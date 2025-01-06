package unit

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/objones25/go-embeddings/pkg/embedding"
	"github.com/stretchr/testify/assert"
)

func setupTestService() embedding.Service {
	// Create a new service instance with test configuration
	config := &embedding.Config{
		ModelPath: filepath.Join("..", "..", "testdata", "model.onnx"),
		Tokenizer: embedding.TokenizerConfig{
			ModelID:        "sentence-transformers/all-MiniLM-L6-v2",
			SequenceLength: 512,
		},
		BatchSize:         4,
		MaxSequenceLength: 512,
		Dimension:         384,
		Options: embedding.Options{
			Normalize:    true,
			CacheEnabled: true,
		},
	}

	svc, err := embedding.NewService(context.Background(), config)
	if err != nil {
		panic(fmt.Sprintf("failed to create test service: %v", err))
	}
	return svc
}

func TestDynamicBatcher(t *testing.T) {
	svc := setupTestService()
	batcher := embedding.NewDynamicBatcher(svc, 32, 500*time.Millisecond, 10*time.Millisecond)
	defer batcher.Shutdown()

	tests := []struct {
		name           string
		numRequests    int
		expectedBatch  int
		window         time.Duration
		timeout        time.Duration
		expectTimeout  bool
		expectError    bool
		errorContains  string
		concurrentReqs bool
	}{
		{
			name:          "single request",
			numRequests:   1,
			expectedBatch: 1,
			window:        10 * time.Millisecond,
			timeout:       500 * time.Millisecond,
		},
		{
			name:          "full batch",
			numRequests:   4,
			expectedBatch: 4,
			window:        10 * time.Millisecond,
			timeout:       500 * time.Millisecond,
		},
		{
			name:          "partial batch with window",
			numRequests:   2,
			expectedBatch: 2,
			window:        5 * time.Millisecond,
			timeout:       500 * time.Millisecond,
		},
		{
			name:           "concurrent requests",
			numRequests:    8,
			expectedBatch:  4,
			window:         10 * time.Millisecond,
			timeout:        2 * time.Second,
			concurrentReqs: true,
		},
		{
			name:          "timeout",
			numRequests:   1,
			expectedBatch: 1,
			window:        10 * time.Millisecond,
			timeout:       1 * time.Millisecond,
			expectTimeout: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			batcher := embedding.NewDynamicBatcher(svc, tt.expectedBatch, tt.timeout, tt.window)
			defer batcher.Shutdown()

			ctx := context.Background()
			if tt.expectTimeout {
				var cancel context.CancelFunc
				ctx, cancel = context.WithTimeout(ctx, tt.timeout)
				defer cancel()
			}

			// Create test requests
			var wg sync.WaitGroup
			results := make([][]float32, tt.numRequests)
			errors := make([]error, tt.numRequests)

			// Function to process a single request
			processRequest := func(i int) {
				defer wg.Done()
				ctx, cancel := context.WithTimeout(context.Background(), tt.timeout)
				defer cancel()
				embedding, err := batcher.QueueRequest(ctx, fmt.Sprintf("Test text %d", i))
				results[i] = embedding
				errors[i] = err
			}

			// Launch requests
			wg.Add(tt.numRequests)
			if tt.concurrentReqs {
				for i := 0; i < tt.numRequests; i++ {
					go processRequest(i)
				}
			} else {
				for i := 0; i < tt.numRequests; i++ {
					processRequest(i)
				}
			}

			// Wait for all requests to complete
			wg.Wait()

			// Verify results
			for i := 0; i < tt.numRequests; i++ {
				if tt.expectTimeout {
					assert.Error(t, errors[i], "expected timeout error")
					assert.Nil(t, results[i], "expected nil result on timeout")
				} else if tt.expectError {
					assert.Error(t, errors[i], "expected error")
					if tt.errorContains != "" {
						assert.Contains(t, errors[i].Error(), tt.errorContains)
					}
				} else {
					assert.NoError(t, errors[i], "unexpected error")
					assert.NotNil(t, results[i], "expected non-nil result")
					assert.Len(t, results[i], 384, "unexpected embedding dimension")
				}
			}
		})
	}
}

func TestDynamicBatcherShutdown(t *testing.T) {
	svc := setupTestService()
	batcher := embedding.NewDynamicBatcher(svc, 4, 100*time.Millisecond, 10*time.Millisecond)

	// Queue some requests
	var wg sync.WaitGroup
	numRequests := 3
	results := make([][]float32, numRequests)
	errors := make([]error, numRequests)
	texts := []string{
		"First test text for shutdown",
		"Second test text for shutdown",
		"Third test text for shutdown",
	}

	for i := 0; i < numRequests; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			embedding, err := batcher.QueueRequest(context.Background(), texts[i])
			results[i] = embedding
			errors[i] = err
		}(i)
	}

	// Wait a bit for requests to be queued
	time.Sleep(5 * time.Millisecond)

	// Shutdown while requests are in flight
	batcher.Shutdown()

	// Verify no deadlock
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		// Success - all requests completed
		shutdownErrors := 0
		successfulRequests := 0

		for i := 0; i < numRequests; i++ {
			if errors[i] != nil {
				if errors[i].Error() == "batcher is shutting down" {
					shutdownErrors++
				} else {
					t.Errorf("unexpected error in request %d: %v", i, errors[i])
				}
			} else {
				successfulRequests++
				assert.NotNil(t, results[i], "expected non-nil result for request %d", i)
				assert.Len(t, results[i], 384, "unexpected embedding dimension for request %d", i)
			}
		}

		// Either all requests should succeed or all should get shutdown errors
		if successfulRequests > 0 && shutdownErrors > 0 {
			t.Errorf("mixed results: %d successful requests and %d shutdown errors", successfulRequests, shutdownErrors)
		}

	case <-time.After(200 * time.Millisecond):
		t.Fatal("shutdown deadlock detected")
	}
}

func TestDynamicBatcherCancellation(t *testing.T) {
	svc := setupTestService()
	batcher := embedding.NewDynamicBatcher(svc, 32, 100*time.Millisecond, 10*time.Millisecond)
	defer batcher.Shutdown()

	// Test context cancellation
	ctx, cancel := context.WithCancel(context.Background())

	// Start a request
	resultCh := make(chan error, 1)
	go func() {
		_, err := batcher.QueueRequest(ctx, "Test text")
		resultCh <- err
	}()

	// Cancel the context immediately
	cancel()

	// Verify the request was cancelled
	select {
	case err := <-resultCh:
		assert.Error(t, err, "expected cancellation error")
		assert.Equal(t, context.Canceled, err)
	case <-time.After(200 * time.Millisecond):
		t.Fatal("cancellation timeout")
	}
}

func TestBatchSizeAutoTuning(t *testing.T) {
	svc := setupTestService()
	// Start with a moderate batch size
	batcher := embedding.NewDynamicBatcher(svc, 4, 500*time.Millisecond, 10*time.Millisecond)
	defer batcher.Shutdown()

	tests := []struct {
		name          string
		inputTexts    []string
		expectedTrend string // "increase", "decrease", or "stable"
		sleepBetween  time.Duration
		numIterations int
		warmupIters   int // Number of warm-up iterations to establish baseline
	}{
		{
			name: "short inputs should increase batch size",
			inputTexts: []string{
				"short text",
				"another short text",
				"third short text",
				"fourth short text",
			},
			expectedTrend: "increase",
			sleepBetween:  300 * time.Millisecond, // Give more time for tuning
			numIterations: 10,                     // More iterations to see the trend
			warmupIters:   3,                      // Warm up iterations to establish baseline
		},
		{
			name: "long inputs should decrease batch size",
			inputTexts: []string{
				strings.Repeat("This is a much longer text that contains multiple sentences and should take longer to process. ", 20),
				strings.Repeat("Another lengthy text that discusses various topics and includes multiple concepts. ", 20),
				strings.Repeat("Yet another long text that contains substantial content and requires significant processing. ", 20),
				strings.Repeat("A fourth long text to ensure we have enough data to process in the batch. ", 20),
			},
			expectedTrend: "decrease",
			sleepBetween:  300 * time.Millisecond,
			numIterations: 10,
			warmupIters:   3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a new batcher for each test to avoid interference
			testBatcher := embedding.NewDynamicBatcher(svc, 4, 500*time.Millisecond, 10*time.Millisecond)
			defer testBatcher.Shutdown()

			// Warm up the batcher to establish baseline
			for i := 0; i < tt.warmupIters; i++ {
				var wg sync.WaitGroup
				for _, text := range tt.inputTexts {
					wg.Add(1)
					go func(text string) {
						defer wg.Done()
						ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
						defer cancel()
						_, err := testBatcher.QueueRequest(ctx, text)
						assert.NoError(t, err, "unexpected error in warm-up request")
					}(text)
				}
				wg.Wait()
				time.Sleep(tt.sleepBetween)
			}

			// Record initial batch size after warm-up
			initialBatchSize := testBatcher.GetCurrentBatchSize()
			prevBatchSize := initialBatchSize
			var batchSizes []int32
			batchSizes = append(batchSizes, initialBatchSize)

			// Run test iterations
			for i := 0; i < tt.numIterations; i++ {
				var wg sync.WaitGroup
				for _, text := range tt.inputTexts {
					wg.Add(1)
					go func(text string) {
						defer wg.Done()
						ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
						defer cancel()
						_, err := testBatcher.QueueRequest(ctx, text)
						assert.NoError(t, err, "unexpected error in request")
					}(text)
				}
				wg.Wait()

				time.Sleep(tt.sleepBetween)

				currentBatchSize := testBatcher.GetCurrentBatchSize()
				batchSizes = append(batchSizes, currentBatchSize)
				t.Logf("Iteration %d: batch size changed from %d to %d", i, prevBatchSize, currentBatchSize)
				prevBatchSize = currentBatchSize
			}

			// Calculate the trend
			increasingCount := 0
			decreasingCount := 0
			for i := 1; i < len(batchSizes); i++ {
				if batchSizes[i] > batchSizes[i-1] {
					increasingCount++
				} else if batchSizes[i] < batchSizes[i-1] {
					decreasingCount++
				}
			}

			t.Logf("Batch size history: %v", batchSizes)
			t.Logf("Increases: %d, Decreases: %d", increasingCount, decreasingCount)

			// Verify the trend with a more lenient threshold
			switch tt.expectedTrend {
			case "increase":
				assert.True(t, increasingCount > decreasingCount || batchSizes[len(batchSizes)-1] > batchSizes[0],
					"expected batch size to show an increasing trend (increases: %d, decreases: %d, initial: %d, final: %d)",
					increasingCount, decreasingCount, batchSizes[0], batchSizes[len(batchSizes)-1])
			case "decrease":
				assert.True(t, decreasingCount > increasingCount || batchSizes[len(batchSizes)-1] < batchSizes[0],
					"expected batch size to show a decreasing trend (decreases: %d, increases: %d, initial: %d, final: %d)",
					decreasingCount, increasingCount, batchSizes[0], batchSizes[len(batchSizes)-1])
			}
		})
	}
}
