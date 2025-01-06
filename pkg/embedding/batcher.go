package embedding

import (
	"context"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"
)

// BatchRequest represents a single request in a batch
type BatchRequest struct {
	Text     string
	Response chan BatchResponse
}

// BatchResponse represents the response for a single request
type BatchResponse struct {
	Embedding []float32
	Error     error
}

// BatcherStats holds performance statistics
type BatcherStats struct {
	totalRequests  atomic.Uint64
	totalBatches   atomic.Uint64
	avgBatchSize   atomic.Int64 // Store as fixed-point number (x1000)
	avgLatency     atomic.Int64 // Store as milliseconds
	avgInputLength atomic.Int64 // Store as fixed-point number (x1000)
	batchSizeHist  [64]atomic.Uint32
	latencyHist    [64]atomic.Uint32
	targetLatency  atomic.Int64 // Target latency in milliseconds
	minBatchSize   atomic.Int32
	maxBatchSize   atomic.Int32
}

// DynamicBatcher handles dynamic batching of embedding requests
type DynamicBatcher struct {
	service      Service
	maxBatchSize atomic.Int32
	timeout      atomic.Int64 // Store duration as int64 nanoseconds
	window       atomic.Int64 // Store duration as int64 nanoseconds
	queue        chan BatchRequest
	doneMu       sync.RWMutex
	done         bool
	activeCount  atomic.Int32
	stats        *BatcherStats
	tuningMu     sync.Mutex
	lastTuning   time.Time
}

// NewDynamicBatcher creates a new dynamic batcher
func NewDynamicBatcher(svc Service, maxBatchSize int, timeout, window time.Duration) *DynamicBatcher {
	if maxBatchSize <= 0 {
		maxBatchSize = 32
	}
	if timeout <= 0 {
		timeout = 100 * time.Millisecond
	}
	if window <= 0 {
		window = 10 * time.Millisecond
	}

	b := &DynamicBatcher{
		service:    svc,
		queue:      make(chan BatchRequest, maxBatchSize*4),
		stats:      &BatcherStats{},
		doneMu:     sync.RWMutex{},
		done:       false,
		lastTuning: time.Now(),
	}

	// Initialize atomic values
	b.maxBatchSize.Store(int32(maxBatchSize))
	b.timeout.Store(timeout.Nanoseconds())
	b.window.Store(window.Nanoseconds())

	// Initialize stats
	b.stats.targetLatency.Store(timeout.Milliseconds() / 2) // Target 50% of timeout
	b.stats.minBatchSize.Store(1)
	b.stats.maxBatchSize.Store(int32(maxBatchSize))

	b.start()
	return b
}

// start begins processing batches
func (b *DynamicBatcher) start() {
	go b.processBatches()
}

// processBatches continuously processes incoming requests in batches
func (b *DynamicBatcher) processBatches() {
	var batch []BatchRequest
	timer := time.NewTimer(b.getWindow())
	defer timer.Stop()

	for {
		b.doneMu.RLock()
		isDone := b.done
		b.doneMu.RUnlock()

		if isDone {
			if len(batch) > 0 {
				b.processBatch(batch)
			}
			return
		}

		select {
		case req := <-b.queue:
			batch = append(batch, req)
			if int32(len(batch)) >= b.maxBatchSize.Load() {
				b.processBatch(batch)
				batch = nil
				timer.Reset(b.getWindow())
			}

		case <-timer.C:
			if len(batch) > 0 {
				b.processBatch(batch)
				batch = nil
			}
			timer.Reset(b.getWindow())
		}
	}
}

// processBatch processes a single batch of requests
func (b *DynamicBatcher) processBatch(batch []BatchRequest) {
	if len(batch) == 0 {
		return
	}

	batchStart := time.Now()
	batchSize := len(batch)

	// Update stats
	b.stats.totalBatches.Add(1)
	b.stats.totalRequests.Add(uint64(batchSize))
	b.updateBatchSizeHistogram(batchSize)

	// Track active requests
	b.activeCount.Add(int32(batchSize))
	defer b.activeCount.Add(-int32(batchSize))

	// Calculate average input length
	var totalLength int
	texts := make([]string, batchSize)
	for i, req := range batch {
		texts[i] = req.Text
		totalLength += len(req.Text)
	}
	avgLength := float64(totalLength) / float64(batchSize)
	b.updateAvgInputLength(avgLength)

	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), b.getTimeout())
	defer cancel()

	// Process batch
	embeddings, err := b.service.BatchEmbed(ctx, texts)

	// Update latency stats and tune batch size
	latency := time.Since(batchStart)
	b.updateLatencyHistogram(latency)
	b.updateAvgLatency(latency)
	b.tuneBatchSize(latency, avgLength)

	// Check if the batcher is shutting down
	b.doneMu.RLock()
	isDone := b.done
	b.doneMu.RUnlock()

	// Distribute results
	if isDone {
		for _, req := range batch {
			req.Response <- BatchResponse{Error: fmt.Errorf("batcher is shutting down")}
		}
		return
	}

	if err != nil {
		for _, req := range batch {
			req.Response <- BatchResponse{Error: err}
		}
		return
	}

	for i, req := range batch {
		if i < len(embeddings) {
			req.Response <- BatchResponse{Embedding: embeddings[i]}
		} else {
			req.Response <- BatchResponse{Error: fmt.Errorf("batch processing error: missing embedding")}
		}
	}

	// Update average batch size
	b.updateAvgBatchSize(float64(batchSize))
}

// tuneBatchSize adjusts the batch size based on latency and input length
func (b *DynamicBatcher) tuneBatchSize(latency time.Duration, avgInputLength float64) {
	b.tuningMu.Lock()
	defer b.tuningMu.Unlock()

	// Only tune every 50ms to avoid oscillation
	if time.Since(b.lastTuning) < 50*time.Millisecond {
		return
	}
	b.lastTuning = time.Now()

	targetLatency := time.Duration(b.stats.targetLatency.Load()) * time.Millisecond
	currentBatchSize := b.maxBatchSize.Load()
	minBatchSize := b.stats.minBatchSize.Load()
	maxBatchSize := b.stats.maxBatchSize.Load()

	// Calculate latency ratio (how far we are from target)
	latencyRatio := float64(latency) / float64(targetLatency)

	// Calculate length ratio with exponential moving average
	avgStoredLength := float64(b.stats.avgInputLength.Load()) / 1000
	if avgStoredLength == 0 {
		avgStoredLength = avgInputLength
	}
	// Use EMA with alpha=0.3 for more responsive length tracking
	avgStoredLength = 0.7*avgStoredLength + 0.3*avgInputLength
	lengthRatio := avgInputLength / avgStoredLength

	// Store the updated average length
	b.stats.avgInputLength.Store(int64(avgStoredLength * 1000))

	// Calculate adjustment factor primarily based on input length
	var adjustmentFactor float64
	switch {
	case lengthRatio > 3.0:
		// Extremely long inputs
		adjustmentFactor = 0.5
	case lengthRatio > 2.0:
		// Very long inputs
		adjustmentFactor = 0.7
	case lengthRatio > 1.5:
		// Moderately long inputs
		adjustmentFactor = 0.8
	case lengthRatio < 0.3:
		// Extremely short inputs
		adjustmentFactor = 1.5
	case lengthRatio < 0.5:
		// Very short inputs
		adjustmentFactor = 1.3
	case lengthRatio < 0.8:
		// Moderately short inputs
		adjustmentFactor = 1.2
	default:
		// Near average length, consider latency
		if latencyRatio > 1.5 {
			adjustmentFactor = 0.9
		} else if latencyRatio < 0.8 {
			adjustmentFactor = 1.1
		} else {
			adjustmentFactor = 1.0
		}
	}

	// For very long inputs, ensure we decrease batch size more aggressively
	if lengthRatio > 2.0 && currentBatchSize > minBatchSize {
		// Force a decrease by at least 30%
		adjustmentFactor = math.Min(adjustmentFactor, 0.7)
	}

	// For very short inputs with good latency, be more aggressive in increasing
	if lengthRatio < 0.5 && latencyRatio < 0.8 && currentBatchSize < maxBatchSize {
		// Increase by at least 20%
		adjustmentFactor = math.Max(adjustmentFactor, 1.2)
	}

	// Apply adjustment
	newBatchSize := int32(float64(currentBatchSize) * adjustmentFactor)

	// Ensure we stay within bounds
	if newBatchSize < minBatchSize {
		newBatchSize = minBatchSize
	}
	if newBatchSize > maxBatchSize {
		newBatchSize = maxBatchSize
	}

	// For small batch sizes, ensure we make at least a unit change when needed
	if adjustmentFactor != 1.0 {
		if currentBatchSize <= 8 {
			// For small batch sizes, make at least a unit change
			if adjustmentFactor > 1.0 && newBatchSize <= currentBatchSize {
				newBatchSize = currentBatchSize + 1
			} else if adjustmentFactor < 1.0 && newBatchSize >= currentBatchSize {
				newBatchSize = currentBatchSize - 1
			}
		} else {
			// For larger batch sizes, use percentage threshold
			minChange := float64(currentBatchSize) * 0.05
			if math.Abs(float64(newBatchSize-currentBatchSize)) < minChange {
				if adjustmentFactor > 1.0 {
					newBatchSize = currentBatchSize + int32(math.Ceil(minChange))
				} else {
					newBatchSize = currentBatchSize - int32(math.Ceil(minChange))
				}
			}
		}
	}

	// Only update if there's a change
	if newBatchSize != currentBatchSize {
		b.maxBatchSize.Store(newBatchSize)
	}
}

// updateAvgInputLength updates the average input length statistic
func (b *DynamicBatcher) updateAvgInputLength(length float64) {
	// Store as fixed-point number (x1000 for 3 decimal precision)
	lengthFixed := int64(length * 1000)
	for {
		old := b.stats.avgInputLength.Load()
		count := b.stats.totalBatches.Load()
		// Fixed-point arithmetic
		new := (old*(int64(count)-1) + lengthFixed) / int64(count)
		if b.stats.avgInputLength.CompareAndSwap(old, new) {
			break
		}
	}
}

// QueueRequest adds a request to the batch queue
func (b *DynamicBatcher) QueueRequest(ctx context.Context, text string) ([]float32, error) {
	b.doneMu.RLock()
	if b.done {
		b.doneMu.RUnlock()
		return nil, fmt.Errorf("batcher is shutting down")
	}
	b.doneMu.RUnlock()

	respChan := make(chan BatchResponse, 1)
	req := BatchRequest{
		Text:     text,
		Response: respChan,
	}

	select {
	case b.queue <- req:
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	select {
	case resp := <-respChan:
		return resp.Embedding, resp.Error
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// Shutdown gracefully shuts down the batcher
func (b *DynamicBatcher) Shutdown() {
	b.doneMu.Lock()
	if !b.done {
		b.done = true
		b.doneMu.Unlock()
		// Wait for active requests to complete
		for b.activeCount.Load() > 0 {
			time.Sleep(time.Millisecond)
		}
		close(b.queue)
	} else {
		b.doneMu.Unlock()
	}
}

// Helper methods for atomic duration access
func (b *DynamicBatcher) getTimeout() time.Duration {
	return time.Duration(b.timeout.Load())
}

func (b *DynamicBatcher) getWindow() time.Duration {
	return time.Duration(b.window.Load())
}

// Stats update methods
func (b *DynamicBatcher) updateAvgBatchSize(size float64) {
	// Store as fixed-point number (x1000 for 3 decimal precision)
	sizeFixed := int64(size * 1000)
	for {
		old := b.stats.avgBatchSize.Load()
		count := b.stats.totalBatches.Load()
		// Fixed-point arithmetic
		new := (old*(int64(count)-1) + sizeFixed) / int64(count)
		if b.stats.avgBatchSize.CompareAndSwap(old, new) {
			break
		}
	}
}

func (b *DynamicBatcher) updateAvgLatency(latency time.Duration) {
	latencyMs := int64(latency.Milliseconds())
	for {
		old := b.stats.avgLatency.Load()
		count := b.stats.totalBatches.Load()
		new := (old*(int64(count)-1) + latencyMs) / int64(count)
		if b.stats.avgLatency.CompareAndSwap(old, new) {
			break
		}
	}
}

func (b *DynamicBatcher) updateBatchSizeHistogram(size int) {
	idx := min(size/8, 63) // 8 requests per bucket, max 64 buckets
	b.stats.batchSizeHist[idx].Add(1)
}

func (b *DynamicBatcher) updateLatencyHistogram(latency time.Duration) {
	idx := min(int(latency.Milliseconds()/10), 63) // 10ms per bucket, max 64 buckets
	b.stats.latencyHist[idx].Add(1)
}

// GetStats returns current batcher statistics
func (b *DynamicBatcher) GetStats() *BatcherStats {
	stats := &BatcherStats{}
	stats.totalRequests.Store(b.stats.totalRequests.Load())
	stats.totalBatches.Store(b.stats.totalBatches.Load())
	stats.avgBatchSize.Store(b.stats.avgBatchSize.Load())
	stats.avgLatency.Store(b.stats.avgLatency.Load())
	stats.avgInputLength.Store(b.stats.avgInputLength.Load())
	stats.targetLatency.Store(b.stats.targetLatency.Load())
	stats.minBatchSize.Store(b.stats.minBatchSize.Load())
	stats.maxBatchSize.Store(b.stats.maxBatchSize.Load())
	return stats
}

// GetCurrentBatchSize returns the current maximum batch size
func (b *DynamicBatcher) GetCurrentBatchSize() int32 {
	return b.maxBatchSize.Load()
}
