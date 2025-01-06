package embedding

import (
	"container/list"
	"sync"
)

// Cache implements a thread-safe LRU cache for embeddings
type Cache struct {
	capacity int
	items    map[string]*list.Element
	lru      *list.List
	mu       sync.RWMutex
}

type cacheItem struct {
	key       string
	embedding []float32
}

// NewCache creates a new cache with the specified capacity
func NewCache(capacity int) *Cache {
	return &Cache{
		capacity: capacity,
		items:    make(map[string]*list.Element),
		lru:      list.New(),
	}
}

// Get retrieves an embedding from the cache
func (c *Cache) Get(key string) ([]float32, bool) {
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

// Set adds or updates an embedding in the cache
func (c *Cache) Set(key string, embedding []float32) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// If key exists, update it
	if element, exists := c.items[key]; exists {
		c.lru.MoveToFront(element)
		item := element.Value.(*cacheItem)
		// Create a copy of the embedding
		newEmbedding := make([]float32, len(embedding))
		copy(newEmbedding, embedding)
		item.embedding = newEmbedding
		return
	}

	// Create a copy of the embedding
	newEmbedding := make([]float32, len(embedding))
	copy(newEmbedding, embedding)

	// Add new item
	item := &cacheItem{
		key:       key,
		embedding: newEmbedding,
	}
	element := c.lru.PushFront(item)
	c.items[key] = element

	// Remove oldest item if cache is full
	if c.lru.Len() > c.capacity {
		c.removeOldest()
	}
}

// Remove removes an embedding from the cache
func (c *Cache) Remove(key string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if element, exists := c.items[key]; exists {
		c.removeElement(element)
	}
}

// Clear removes all items from the cache
func (c *Cache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.items = make(map[string]*list.Element)
	c.lru.Init()
}

// Len returns the number of items in the cache
func (c *Cache) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.items)
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
}
