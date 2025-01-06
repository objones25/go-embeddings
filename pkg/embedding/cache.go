package embedding

import (
	"container/heap"
	"container/list"
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"
)

// Cache implements a thread-safe multi-level (memory + disk) LRU cache for embeddings
type Cache struct {
	capacity     int          // maximum number of items in memory
	maxSizeBytes int64        // maximum size in bytes for memory cache
	currentSize  atomic.Int64 // current size in bytes
	items        map[string]*list.Element
	lru          *list.List
	mu           sync.RWMutex

	// Disk cache settings
	diskEnabled   bool
	diskCachePath string
	diskMu        sync.RWMutex

	// Cache warming
	accessCount    map[string]*AccessStats
	accessCountMu  sync.RWMutex
	warmerActive   atomic.Bool
	warmerInterval time.Duration
	warmerCtx      context.Context
	warmerCancel   context.CancelFunc

	// Cache metrics
	hits   atomic.Int64
	misses atomic.Int64
}

// cacheItem represents a single item in the cache
type cacheItem struct {
	key       string
	embedding []float32
	sizeBytes int64
}

// AccessStats tracks access patterns for cache items
type AccessStats struct {
	count    int64     // number of accesses
	lastUsed time.Time // last access time
}

// CacheOptions configures the cache behavior
type CacheOptions struct {
	Capacity       int
	MaxSizeBytes   int64
	DiskEnabled    bool
	CachePath      string
	WarmerEnabled  bool
	WarmerInterval time.Duration
}

// NewCache creates a new cache with the specified options
func NewCache(opts CacheOptions) (*Cache, error) {
	if opts.MaxSizeBytes <= 0 {
		opts.MaxSizeBytes = 1 << 30 // 1GB default
	}
	if opts.Capacity <= 0 {
		opts.Capacity = 10000 // 10k items default
	}
	if opts.WarmerInterval <= 0 {
		opts.WarmerInterval = 5 * time.Minute // 5 minutes default
	}

	ctx, cancel := context.WithCancel(context.Background())

	c := &Cache{
		capacity:       opts.Capacity,
		maxSizeBytes:   opts.MaxSizeBytes,
		items:          make(map[string]*list.Element),
		lru:            list.New(),
		diskEnabled:    opts.DiskEnabled,
		accessCount:    make(map[string]*AccessStats),
		warmerInterval: opts.WarmerInterval,
		warmerCtx:      ctx,
		warmerCancel:   cancel,
	}

	if opts.DiskEnabled {
		if opts.CachePath == "" {
			opts.CachePath = filepath.Join(os.TempDir(), "embeddings-cache")
		}
		if err := os.MkdirAll(opts.CachePath, 0755); err != nil {
			return nil, fmt.Errorf("failed to create cache directory: %v", err)
		}
		c.diskCachePath = opts.CachePath
	}

	if opts.WarmerEnabled {
		c.startWarmer()
	}

	return c, nil
}

// Get retrieves an embedding from the cache (memory first, then disk)
func (c *Cache) Get(key string) ([]float32, bool) {
	// Update access stats
	c.updateAccessStats(key)

	// Try memory cache first
	if embedding, found := c.getFromMemory(key); found {
		c.hits.Add(1)
		return embedding, true
	}

	// Try disk cache if enabled
	if c.diskEnabled {
		if embedding, found := c.getFromDisk(key); found {
			// Promote to memory cache
			c.Set(key, embedding)
			c.hits.Add(1)
			return embedding, true
		}
	}

	c.misses.Add(1)
	return nil, false
}

// updateAccessStats updates the access statistics for a key
func (c *Cache) updateAccessStats(key string) {
	c.accessCountMu.Lock()
	defer c.accessCountMu.Unlock()

	stats, exists := c.accessCount[key]
	if !exists {
		stats = &AccessStats{}
		c.accessCount[key] = stats
	}
	stats.count++
	stats.lastUsed = time.Now()
}

// startWarmer starts the cache warming background worker
func (c *Cache) startWarmer() {
	c.warmerActive.Store(true)
	go c.warmerWorker()
}

// stopWarmer stops the cache warming background worker
func (c *Cache) stopWarmer() {
	if c.warmerActive.Load() {
		c.warmerCancel()
		c.warmerActive.Store(false)
	}
}

// warmerWorker runs the cache warming logic
func (c *Cache) warmerWorker() {
	ticker := time.NewTicker(c.warmerInterval)
	defer ticker.Stop()

	for {
		select {
		case <-c.warmerCtx.Done():
			return
		case <-ticker.C:
			c.warmCache()
		}
	}
}

// warmCache analyzes access patterns and pre-warms frequently accessed items
func (c *Cache) warmCache() {
	// Get current memory cache capacity
	c.mu.RLock()
	currentItems := c.lru.Len()
	c.mu.RUnlock()

	// Calculate how many items we can warm
	// Leave 20% of cache capacity for new items
	maxWarmItems := (c.capacity * 80) / 100
	if currentItems >= maxWarmItems {
		return
	}

	// Get top accessed items
	topItems := c.getTopAccessedItems(maxWarmItems - currentItems)

	// Pre-warm items
	for _, item := range topItems {
		if embedding, found := c.getFromDisk(item.key); found {
			c.Set(item.key, embedding)
		}
	}
}

// accessItem represents an item with its access statistics
type accessItem struct {
	key   string
	stats *AccessStats
}

// accessHeap implements heap.Interface for sorting by access count and recency
type accessHeap []accessItem

func (h accessHeap) Len() int { return len(h) }
func (h accessHeap) Less(i, j int) bool {
	// Sort by access count first, then by recency
	if h[i].stats.count != h[j].stats.count {
		return h[i].stats.count > h[j].stats.count
	}
	return h[i].stats.lastUsed.After(h[j].stats.lastUsed)
}
func (h accessHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *accessHeap) Push(x interface{}) { *h = append(*h, x.(accessItem)) }
func (h *accessHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[0 : n-1]
	return item
}

// getTopAccessedItems returns the top N most accessed items
func (c *Cache) getTopAccessedItems(n int) []accessItem {
	c.accessCountMu.RLock()
	defer c.accessCountMu.RUnlock()

	// Create a min-heap of the top N items
	h := &accessHeap{}
	heap.Init(h)

	for key, stats := range c.accessCount {
		// Skip items already in memory cache
		if _, found := c.getFromMemory(key); found {
			continue
		}

		item := accessItem{key: key, stats: stats}
		if h.Len() < n {
			heap.Push(h, item)
		} else if (*h)[0].stats.count < stats.count {
			heap.Pop(h)
			heap.Push(h, item)
		}
	}

	// Convert heap to sorted slice
	result := make([]accessItem, h.Len())
	for i := len(result) - 1; i >= 0; i-- {
		result[i] = heap.Pop(h).(accessItem)
	}

	return result
}

// Close cleans up cache resources
func (c *Cache) Close() error {
	c.stopWarmer()
	return c.Clear()
}

// getFromMemory retrieves an embedding from memory cache
func (c *Cache) getFromMemory(key string) ([]float32, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if element, exists := c.items[key]; exists {
		c.lru.MoveToFront(element)
		item := element.Value.(*cacheItem)
		// Return a copy to prevent modification of cached data
		embedding := make([]float32, len(item.embedding))
		copy(embedding, item.embedding)
		return embedding, true
	}
	return nil, false
}

// getFromDisk retrieves an embedding from disk cache
func (c *Cache) getFromDisk(key string) ([]float32, bool) {
	c.diskMu.RLock()
	defer c.diskMu.RUnlock()

	path := filepath.Join(c.diskCachePath, fmt.Sprintf("%x.bin", stringHash(key)))
	file, err := os.Open(path)
	if err != nil {
		return nil, false
	}
	defer file.Close()

	// Read embedding length
	var length int32
	if err := binary.Read(file, binary.LittleEndian, &length); err != nil {
		return nil, false
	}

	// Read embedding data
	embedding := make([]float32, length)
	if err := binary.Read(file, binary.LittleEndian, embedding); err != nil {
		return nil, false
	}

	return embedding, true
}

// Set adds or updates an embedding in both memory and disk cache
func (c *Cache) Set(key string, embedding []float32) error {
	if err := c.setInMemory(key, embedding); err != nil {
		return err
	}

	if c.diskEnabled {
		return c.setOnDisk(key, embedding)
	}

	return nil
}

// setInMemory adds or updates an embedding in memory cache
func (c *Cache) setInMemory(key string, embedding []float32) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	newSize := calculateSize(embedding)

	// If key exists, update it
	if element, exists := c.items[key]; exists {
		c.lru.MoveToFront(element)
		item := element.Value.(*cacheItem)
		oldSize := item.sizeBytes
		// Create a copy of the embedding
		newEmbedding := make([]float32, len(embedding))
		copy(newEmbedding, embedding)
		item.embedding = newEmbedding
		item.sizeBytes = newSize
		c.currentSize.Add(newSize - oldSize)
		return nil
	}

	// Remove items if we're over capacity or size limit
	for c.lru.Len() >= c.capacity || c.currentSize.Load()+newSize > c.maxSizeBytes {
		if c.lru.Len() == 0 {
			return fmt.Errorf("item too large for cache")
		}
		c.removeOldest()
	}

	// Create a copy of the embedding
	newEmbedding := make([]float32, len(embedding))
	copy(newEmbedding, embedding)

	// Add new item
	item := &cacheItem{
		key:       key,
		embedding: newEmbedding,
		sizeBytes: newSize,
	}
	element := c.lru.PushFront(item)
	c.items[key] = element
	c.currentSize.Add(newSize)
	return nil
}

// setOnDisk adds or updates an embedding in disk cache
func (c *Cache) setOnDisk(key string, embedding []float32) error {
	c.diskMu.Lock()
	defer c.diskMu.Unlock()

	path := filepath.Join(c.diskCachePath, fmt.Sprintf("%x.bin", stringHash(key)))
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create cache file: %v", err)
	}
	defer file.Close()

	// Write embedding length
	length := int32(len(embedding))
	if err := binary.Write(file, binary.LittleEndian, length); err != nil {
		return fmt.Errorf("failed to write embedding length: %v", err)
	}

	// Write embedding data
	if err := binary.Write(file, binary.LittleEndian, embedding); err != nil {
		return fmt.Errorf("failed to write embedding data: %v", err)
	}

	return nil
}

// Remove removes an embedding from both memory and disk cache
func (c *Cache) Remove(key string) error {
	c.mu.Lock()
	if element, exists := c.items[key]; exists {
		c.removeElement(element)
	}
	c.mu.Unlock()

	if c.diskEnabled {
		c.diskMu.Lock()
		defer c.diskMu.Unlock()

		path := filepath.Join(c.diskCachePath, fmt.Sprintf("%x.bin", stringHash(key)))
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("failed to remove cache file: %v", err)
		}
	}

	return nil
}

// Clear removes all items from both memory and disk cache
func (c *Cache) Clear() error {
	c.mu.Lock()
	c.items = make(map[string]*list.Element)
	c.lru = list.New()
	c.currentSize.Store(0)
	c.hits.Store(0)
	c.misses.Store(0)
	c.mu.Unlock()

	// Clear disk cache if enabled
	if c.diskEnabled && c.diskCachePath != "" {
		c.diskMu.Lock()
		defer c.diskMu.Unlock()

		// Remove the entire cache directory and recreate it
		if err := os.RemoveAll(c.diskCachePath); err != nil {
			return fmt.Errorf("failed to remove cache directory: %v", err)
		}
		if err := os.MkdirAll(c.diskCachePath, 0755); err != nil {
			return fmt.Errorf("failed to recreate cache directory: %v", err)
		}
	}

	return nil
}

// stringHash generates a simple hash of a string
func stringHash(s string) uint64 {
	h := uint64(0)
	for i := 0; i < len(s); i++ {
		h = h*31 + uint64(s[i])
	}
	return h
}

// calculateSize calculates the size of an embedding in bytes
func calculateSize(embedding []float32) int64 {
	return int64(len(embedding) * 4) // 4 bytes per float32
}

// Len returns the number of items in the cache
func (c *Cache) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.items)
}

// Size returns the current size of the cache in bytes
func (c *Cache) Size() int64 {
	return c.currentSize.Load()
}

// removeOldest removes the least recently used item from the cache
func (c *Cache) removeOldest() {
	if element := c.lru.Back(); element != nil {
		c.removeElement(element)
	}
}

// removeElement removes an element from the cache
func (c *Cache) removeElement(element *list.Element) {
	c.lru.Remove(element)
	item := element.Value.(*cacheItem)
	delete(c.items, item.key)
	c.currentSize.Add(-item.sizeBytes)
}

// GetMetrics returns the current cache hit/miss statistics
func (c *Cache) GetMetrics() CacheMetrics {
	return CacheMetrics{
		Hits:   c.hits.Load(),
		Misses: c.misses.Load(),
	}
}
