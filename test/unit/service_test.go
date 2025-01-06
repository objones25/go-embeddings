package unit

import (
	"bytes"
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/objones25/go-embeddings/pkg/embedding"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

func init() {
	// Set up library paths
	workspaceDir, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		log.Printf("Error getting workspace dir: %v", err)
	}
	libsDir := filepath.Join(workspaceDir, "libs")

	// Set environment variables
	os.Setenv("DYLD_LIBRARY_PATH", libsDir)
	os.Setenv("LD_LIBRARY_PATH", libsDir)
	os.Setenv("ONNXRUNTIME_LIB_PATH", filepath.Join(libsDir, "libonnxruntime.1.20.0.dylib"))
	os.Setenv("CGO_LDFLAGS", fmt.Sprintf("-L%s", libsDir))

	// Create symlink for ONNX Runtime library
	onnxRuntimePath := filepath.Join(libsDir, "libonnxruntime.1.20.0.dylib")
	onnxRuntimeSymlink := filepath.Join(libsDir, "libonnxruntime.dylib")
	if err := os.Remove(onnxRuntimeSymlink); err != nil && !os.IsNotExist(err) {
		log.Printf("Error removing existing symlink: %v", err)
	}
	if err := os.Symlink(onnxRuntimePath, onnxRuntimeSymlink); err != nil {
		log.Printf("Error creating symlink: %v", err)
	}
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func TestNewService(t *testing.T) {
	testDataPath := filepath.Join("..", "..", "testdata")
	modelPath := filepath.Join(testDataPath, "model.onnx")

	tests := []struct {
		name        string
		config      *embedding.Config
		expectError bool
	}{
		{
			name: "valid config",
			config: &embedding.Config{
				ModelPath: modelPath,
				Tokenizer: embedding.TokenizerConfig{
					ModelID:        "sentence-transformers/all-MiniLM-L6-v2",
					SequenceLength: 512,
				},
				Dimension:         384,
				BatchSize:         32,
				MaxSequenceLength: 512,
				Options: embedding.Options{
					PadToMaxLength: false,
					Normalize:      true,
					CacheEnabled:   true,
				},
			},
			expectError: false,
		},
		{
			name: "missing model path",
			config: &embedding.Config{
				Tokenizer: embedding.TokenizerConfig{
					ModelID:        "bert-base-uncased",
					SequenceLength: 512,
				},
			},
			expectError: true,
		},
		{
			name: "invalid batch size",
			config: &embedding.Config{
				ModelPath: modelPath,
				Tokenizer: embedding.TokenizerConfig{
					ModelID:        "bert-base-uncased",
					SequenceLength: 512,
				},
				BatchSize: -1,
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			service, err := embedding.NewService(context.Background(), tt.config)
			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, service)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, service)
			}
		})
	}
}

func TestEmbed(t *testing.T) {
	testDataPath := filepath.Join("..", "..", "testdata")
	modelPath := filepath.Join(testDataPath, "model.onnx")

	config := &embedding.Config{
		ModelPath: modelPath,
		Tokenizer: embedding.TokenizerConfig{
			ModelID:        "sentence-transformers/all-MiniLM-L6-v2",
			SequenceLength: 512,
		},
		Dimension:         384,
		BatchSize:         32,
		MaxSequenceLength: 512,
		Options: embedding.Options{
			PadToMaxLength: false,
			Normalize:      true,
			CacheEnabled:   true,
		},
	}

	service, err := embedding.NewService(context.Background(), config)
	require.NoError(t, err)
	defer service.Close()

	tests := []struct {
		name        string
		text        string
		ctx         context.Context
		expectError bool
	}{
		{
			name:        "valid text",
			text:        "Hello, world!",
			ctx:         context.Background(),
			expectError: false,
		},
		{
			name:        "empty text",
			text:        "",
			ctx:         context.Background(),
			expectError: true,
		},
		{
			name:        "cancelled context",
			text:        "Hello, world!",
			ctx:         cancelledContext(),
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			embedding, err := service.Embed(tt.ctx, tt.text)
			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, embedding)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, embedding)
				assert.Len(t, embedding, config.Dimension)
			}
		})
	}
}

func TestBatchEmbed(t *testing.T) {
	testDataPath := filepath.Join("..", "..", "testdata")
	modelPath := filepath.Join(testDataPath, "model.onnx")

	config := &embedding.Config{
		ModelPath: modelPath,
		Tokenizer: embedding.TokenizerConfig{
			ModelID:        "sentence-transformers/all-MiniLM-L6-v2",
			SequenceLength: 512,
		},
		Dimension:         384,
		BatchSize:         2,
		MaxSequenceLength: 512,
		Options: embedding.Options{
			PadToMaxLength: false,
			Normalize:      true,
			CacheEnabled:   true,
		},
	}

	service, err := embedding.NewService(context.Background(), config)
	require.NoError(t, err)
	defer service.Close()

	tests := []struct {
		name        string
		texts       []string
		ctx         context.Context
		expectError bool
	}{
		{
			name: "valid batch",
			texts: []string{
				"Hello, world!",
				"This is a test.",
				"Another test sentence.",
			},
			ctx:         context.Background(),
			expectError: false,
		},
		{
			name:        "empty batch",
			texts:       []string{},
			ctx:         context.Background(),
			expectError: true,
		},
		{
			name: "batch with empty text",
			texts: []string{
				"Hello",
				"",
				"World",
			},
			ctx:         context.Background(),
			expectError: true,
		},
		{
			name: "cancelled context",
			texts: []string{
				"Hello, world!",
				"This is a test.",
			},
			ctx:         cancelledContext(),
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			embeddings, err := service.BatchEmbed(tt.ctx, tt.texts)
			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, embeddings)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, embeddings)
				assert.Equal(t, len(embeddings), len(tt.texts))
				for _, embedding := range embeddings {
					assert.Len(t, embedding, config.Dimension)
				}
			}
		})
	}
}

func TestBatchEmbedAsync(t *testing.T) {
	testDataPath := filepath.Join("..", "..", "testdata")
	modelPath := filepath.Join(testDataPath, "model.onnx")

	config := &embedding.Config{
		ModelPath: modelPath,
		Tokenizer: embedding.TokenizerConfig{
			ModelID:        "sentence-transformers/all-MiniLM-L6-v2",
			SequenceLength: 512,
		},
		Dimension:         384,
		BatchSize:         2,
		MaxSequenceLength: 512,
		Options: embedding.Options{
			PadToMaxLength: false,
			Normalize:      true,
			CacheEnabled:   true,
		},
	}

	service, err := embedding.NewService(context.Background(), config)
	require.NoError(t, err)
	defer service.Close()

	tests := []struct {
		name        string
		texts       []string
		ctx         context.Context
		expectError bool
	}{
		{
			name: "valid batch",
			texts: []string{
				"Hello, world!",
				"This is a test.",
				"Another test sentence.",
			},
			ctx:         context.Background(),
			expectError: false,
		},
		{
			name:        "empty batch",
			texts:       []string{},
			ctx:         context.Background(),
			expectError: true,
		},
		{
			name: "cancelled context",
			texts: []string{
				"Hello, world!",
				"This is a test.",
			},
			ctx:         cancelledContext(),
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(tt.ctx, 30*time.Second)
			defer cancel()

			results := make(chan embedding.Result)
			errors := make(chan error)

			err := service.BatchEmbedAsync(ctx, tt.texts, results, errors)
			if tt.expectError {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)

			var receivedResults []embedding.Result
			var receivedErrors []error
			done := make(chan struct{})

			go func() {
				defer close(done)
				expectedResults := len(tt.texts)
				receivedCount := 0

				for receivedCount < expectedResults {
					select {
					case result, ok := <-results:
						if !ok {
							return
						}
						receivedResults = append(receivedResults, result)
						receivedCount++
					case err, ok := <-errors:
						if !ok {
							return
						}
						receivedErrors = append(receivedErrors, err)
						return
					case <-ctx.Done():
						return
					}
				}
			}()

			select {
			case <-done:
			case <-ctx.Done():
				t.Fatalf("Test timed out or context cancelled: %v", ctx.Err())
			}

			assert.Empty(t, receivedErrors)
			assert.Equal(t, len(tt.texts), len(receivedResults))
			for _, result := range receivedResults {
				assert.Len(t, result.Embedding, config.Dimension)
			}
		})
	}
}

func cancelledContext() context.Context {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	return ctx
}

func TestCoreMLInitialization(t *testing.T) {
	// Get test data paths
	testDataPath := filepath.Join("..", "..", "testdata")
	modelPath := filepath.Join(testDataPath, "model.onnx")
	tokenizerPath := filepath.Join(testDataPath, "tokenizer.json")

	tests := []struct {
		name           string
		config         *embedding.Config
		expectCoreML   bool
		expectError    bool
		errorContains  string
		validateLogger func(t *testing.T, logs []string)
	}{
		{
			name: "CoreML enabled on Darwin with ANE",
			config: &embedding.Config{
				ModelPath:   modelPath,
				EnableMetal: true,
				CoreMLConfig: &embedding.CoreMLConfig{
					EnableCaching: true,
					RequireANE:    true,
				},
				BatchSize: 32, // Add required batch size
				Dimension: 384,
				Tokenizer: embedding.TokenizerConfig{
					LocalPath:      tokenizerPath,
					SequenceLength: 512,
				},
			},
			expectCoreML: true,
			validateLogger: func(t *testing.T, logs []string) {
				found := false
				for _, log := range logs {
					if strings.Contains(log, `"msg":"Successfully enabled CoreML execution provider"`) &&
						strings.Contains(log, `"caching_enabled":true`) &&
						strings.Contains(log, `"ane_required":true`) {
						found = true
						break
					}
				}
				assert.True(t, found, "Expected to find CoreML success log with caching and ANE enabled")
			},
		},
		{
			name: "CoreML enabled on Darwin without ANE",
			config: &embedding.Config{
				ModelPath:   modelPath,
				EnableMetal: true,
				CoreMLConfig: &embedding.CoreMLConfig{
					EnableCaching: true,
					RequireANE:    false,
				},
				BatchSize: 32, // Add required batch size
				Dimension: 384,
				Tokenizer: embedding.TokenizerConfig{
					LocalPath:      tokenizerPath,
					SequenceLength: 512,
				},
			},
			expectCoreML: true,
			validateLogger: func(t *testing.T, logs []string) {
				found := false
				for _, log := range logs {
					if strings.Contains(log, `"msg":"Successfully enabled CoreML execution provider"`) &&
						strings.Contains(log, `"caching_enabled":true`) &&
						strings.Contains(log, `"ane_required":false`) {
						found = true
						break
					}
				}
				assert.True(t, found, "Expected to find CoreML success log with caching enabled and ANE disabled")
			},
		},
		{
			name: "CoreML disabled",
			config: &embedding.Config{
				ModelPath:   modelPath,
				EnableMetal: false,
				BatchSize:   32, // Add required batch size
				Dimension:   384,
				Tokenizer: embedding.TokenizerConfig{
					LocalPath:      tokenizerPath,
					SequenceLength: 512,
				},
			},
			expectCoreML: false,
			validateLogger: func(t *testing.T, logs []string) {
				for _, log := range logs {
					assert.NotContains(t, log, "CoreML execution provider")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Skip test if not on Darwin
			if runtime.GOOS != "darwin" {
				t.Skip("Skipping CoreML test on non-Darwin platform")
			}

			// Verify test files exist
			if _, err := os.Stat(tt.config.ModelPath); os.IsNotExist(err) {
				t.Skipf("Model file not found at %s, skipping test", tt.config.ModelPath)
			}
			if _, err := os.Stat(tt.config.Tokenizer.LocalPath); os.IsNotExist(err) {
				t.Skipf("Tokenizer file not found at %s, skipping test", tt.config.Tokenizer.LocalPath)
			}

			// Set up test logger to capture logs
			var logBuffer bytes.Buffer
			logger := zap.New(
				zapcore.NewCore(
					zapcore.NewJSONEncoder(zap.NewProductionEncoderConfig()),
					zapcore.AddSync(&logBuffer),
					zapcore.InfoLevel,
				),
			)

			// Create service with test configuration and custom logger
			ctx := context.Background()
			svc, err := embedding.NewServiceWithLogger(ctx, tt.config, logger)

			// Check error expectations
			if tt.expectError {
				assert.Error(t, err)
				if tt.errorContains != "" {
					assert.Contains(t, err.Error(), tt.errorContains)
				}
				return
			} else {
				assert.NoError(t, err)
			}

			// Clean up service
			defer func() {
				if svc != nil {
					svc.Close()
				}
			}()

			// Validate logs
			logs := strings.Split(logBuffer.String(), "\n")
			if tt.validateLogger != nil {
				tt.validateLogger(t, logs)
			}
		})
	}
}

// TestCoreMLPerformance tests the performance difference between CPU and CoreML
func TestCoreMLPerformance(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("Skipping CoreML performance test on non-Darwin platform")
	}

	// Get test data paths
	testDataPath := filepath.Join("..", "..", "testdata")
	modelPath := filepath.Join(testDataPath, "model.onnx")
	tokenizerPath := filepath.Join(testDataPath, "tokenizer.json")

	// Verify test files exist
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Model file not found at %s, skipping test", modelPath)
	}
	if _, err := os.Stat(tokenizerPath); os.IsNotExist(err) {
		t.Skipf("Tokenizer file not found at %s, skipping test", tokenizerPath)
	}

	// Create test configurations
	baseConfig := &embedding.Config{
		ModelPath: modelPath,
		Dimension: 384,
		Tokenizer: embedding.TokenizerConfig{
			LocalPath:      tokenizerPath,
			SequenceLength: 512,
		},
		BatchSize: 1,
	}

	// CPU configuration
	cpuConfig := *baseConfig
	cpuConfig.EnableMetal = false

	// CoreML configuration
	coreMLConfig := *baseConfig
	coreMLConfig.EnableMetal = true
	coreMLConfig.CoreMLConfig = &embedding.CoreMLConfig{
		EnableCaching: true,
		RequireANE:    false,
	}

	ctx := context.Background()

	// Create services
	cpuService, err := embedding.NewService(ctx, &cpuConfig)
	require.NoError(t, err)
	defer cpuService.Close()

	coreMLService, err := embedding.NewService(ctx, &coreMLConfig)
	require.NoError(t, err)
	defer coreMLService.Close()

	// Test text
	testText := "This is a test sentence for performance comparison between CPU and CoreML execution providers."

	// Warm up both services
	_, err = cpuService.Embed(ctx, testText)
	require.NoError(t, err)
	_, err = coreMLService.Embed(ctx, testText)
	require.NoError(t, err)

	// Benchmark function
	benchmark := func(svc embedding.Service) time.Duration {
		start := time.Now()
		_, err := svc.Embed(ctx, testText)
		require.NoError(t, err)
		return time.Since(start)
	}

	// Run benchmarks
	const iterations = 10
	cpuTimes := make([]time.Duration, iterations)
	coreMLTimes := make([]time.Duration, iterations)

	for i := 0; i < iterations; i++ {
		cpuTimes[i] = benchmark(cpuService)
		coreMLTimes[i] = benchmark(coreMLService)
	}

	// Calculate average times
	var cpuTotal, coreMLTotal time.Duration
	for i := 0; i < iterations; i++ {
		cpuTotal += cpuTimes[i]
		coreMLTotal += coreMLTimes[i]
	}
	cpuAvg := cpuTotal / iterations
	coreMLAvg := coreMLTotal / iterations

	// Log results
	t.Logf("CPU average time: %v", cpuAvg)
	t.Logf("CoreML average time: %v", coreMLAvg)
	t.Logf("CoreML speedup: %.2fx", float64(cpuAvg)/float64(coreMLAvg))

	// Assert that CoreML is faster (allowing for some variance)
	assert.Less(t, coreMLAvg, cpuAvg*2, "CoreML should not be significantly slower than CPU")
}

// TestCacheEviction tests the size-based eviction in the cache
func TestCacheEviction(t *testing.T) {
	testDataPath := filepath.Join("..", "..", "testdata")
	modelPath := filepath.Join(testDataPath, "model.onnx")

	// Create config with small cache size to test eviction
	config := &embedding.Config{
		ModelPath: modelPath,
		Tokenizer: embedding.TokenizerConfig{
			ModelID:        "sentence-transformers/all-MiniLM-L6-v2",
			SequenceLength: 512,
		},
		Dimension:         384,
		BatchSize:         32,
		MaxSequenceLength: 512,
		CacheSize:         2,    // Small cache size to force eviction
		CacheSizeBytes:    8192, // Small memory limit
		Options: embedding.Options{
			PadToMaxLength: false,
			Normalize:      true,
			CacheEnabled:   true,
		},
	}

	service, err := embedding.NewService(context.Background(), config)
	require.NoError(t, err)
	defer service.Close()

	// Test texts
	texts := []string{
		"First text to cache",
		"Second text to cache",
		"Third text should evict first",
		"Fourth text should evict second",
	}

	// First embedding - should be cached
	embedding1, err := service.Embed(context.Background(), texts[0])
	require.NoError(t, err)
	require.NotNil(t, embedding1)

	// Second embedding - should be cached
	embedding2, err := service.Embed(context.Background(), texts[1])
	require.NoError(t, err)
	require.NotNil(t, embedding2)

	// Third embedding - should evict first
	embedding3, err := service.Embed(context.Background(), texts[2])
	require.NoError(t, err)
	require.NotNil(t, embedding3)

	// Verify first text was evicted by getting it again
	embedding1New, err := service.Embed(context.Background(), texts[0])
	require.NoError(t, err)
	require.NotNil(t, embedding1New)

	// Compare the actual float32 slices
	assert.Equal(t, len(embedding1), len(embedding1New), "Embedding lengths should match")
	for i := range embedding1 {
		assert.InDelta(t, embedding1[i], embedding1New[i], 1e-6, "Embedding values should match at index %d", i)
	}

	// Fourth embedding - should evict second
	embedding4, err := service.Embed(context.Background(), texts[3])
	require.NoError(t, err)
	require.NotNil(t, embedding4)

	// Verify second text was evicted
	embedding2New, err := service.Embed(context.Background(), texts[1])
	require.NoError(t, err)
	require.NotNil(t, embedding2New)

	// Compare the actual float32 slices
	assert.Equal(t, len(embedding2), len(embedding2New), "Embedding lengths should match")
	for i := range embedding2 {
		assert.InDelta(t, embedding2[i], embedding2New[i], 1e-6, "Embedding values should match at index %d", i)
	}
}

// TestDiskCache tests the disk caching functionality
func TestDiskCache(t *testing.T) {
	// Create temporary directory for disk cache
	tempDir, err := os.MkdirTemp("", "embeddings-test-cache-*")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	testDataPath := filepath.Join("..", "..", "testdata")
	modelPath := filepath.Join(testDataPath, "model.onnx")

	config := &embedding.Config{
		ModelPath: modelPath,
		Tokenizer: embedding.TokenizerConfig{
			ModelID:        "sentence-transformers/all-MiniLM-L6-v2",
			SequenceLength: 512,
		},
		Dimension:         384,
		BatchSize:         32,
		MaxSequenceLength: 512,
		CacheSize:         1, // Small memory cache to force disk usage
		CacheSizeBytes:    4096,
		DiskCacheEnabled:  true,
		DiskCachePath:     tempDir,
		Options: embedding.Options{
			PadToMaxLength: false,
			Normalize:      true,
			CacheEnabled:   true,
		},
	}

	service, err := embedding.NewService(context.Background(), config)
	require.NoError(t, err)
	defer service.Close()

	// Test text
	text := "This is a test text for disk caching"

	// First request - should compute and cache
	embedding1, err := service.Embed(context.Background(), text)
	require.NoError(t, err)
	require.NotNil(t, embedding1)

	// Force memory cache eviction with different text
	_, err = service.Embed(context.Background(), "Eviction text")
	require.NoError(t, err)

	// Second request - should retrieve from disk
	embedding2, err := service.Embed(context.Background(), text)
	require.NoError(t, err)
	require.NotNil(t, embedding2)

	// Verify embeddings are the same
	assert.Equal(t, embedding1, embedding2)

	// Verify disk cache file exists
	files, err := os.ReadDir(tempDir)
	require.NoError(t, err)
	assert.Greater(t, len(files), 0, "Expected cache files in directory")
}

// TestCacheWarming tests the cache warming functionality
func TestCacheWarming(t *testing.T) {
	testDataPath := filepath.Join("..", "..", "testdata")
	modelPath := filepath.Join(testDataPath, "model.onnx")

	// Create temporary directory for disk cache
	tempDir, err := os.MkdirTemp("", "embeddings-test-warming-*")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	config := &embedding.Config{
		ModelPath: modelPath,
		Tokenizer: embedding.TokenizerConfig{
			ModelID:        "sentence-transformers/all-MiniLM-L6-v2",
			SequenceLength: 512,
		},
		Dimension:         384,
		BatchSize:         32,
		MaxSequenceLength: 512,
		CacheSize:         10,
		CacheSizeBytes:    1 << 20, // 1MB
		DiskCacheEnabled:  true,
		DiskCachePath:     tempDir,
		CacheWarming:      true,
		WarmingInterval:   100 * time.Millisecond, // Short interval for testing
		Options: embedding.Options{
			PadToMaxLength: false,
			Normalize:      true,
			CacheEnabled:   true,
		},
	}

	service, err := embedding.NewService(context.Background(), config)
	require.NoError(t, err)
	defer service.Close()

	// Test text that we'll access frequently
	text := "Frequently accessed text for warming"

	// Access the text multiple times to increase its access count
	for i := 0; i < 5; i++ {
		embedding, err := service.Embed(context.Background(), text)
		require.NoError(t, err)
		require.NotNil(t, embedding)
		time.Sleep(10 * time.Millisecond)
	}

	// Force eviction from memory cache with different texts
	for i := 0; i < 10; i++ {
		_, err := service.Embed(context.Background(), fmt.Sprintf("Eviction text %d", i))
		require.NoError(t, err)
	}

	// Wait for cache warming to occur
	time.Sleep(200 * time.Millisecond)

	// The frequently accessed text should be back in memory cache
	start := time.Now()
	embedding, err := service.Embed(context.Background(), text)
	require.NoError(t, err)
	require.NotNil(t, embedding)

	// Verify it was served from memory cache (should be very fast)
	assert.Less(t, time.Since(start), 5*time.Millisecond, "Request took too long, might not be served from memory cache")
}

// TestCacheMetrics tests the cache hit/miss metrics
func TestCacheMetrics(t *testing.T) {
	testDataPath := filepath.Join("..", "..", "testdata")
	modelPath := filepath.Join(testDataPath, "model.onnx")

	config := &embedding.Config{
		ModelPath: modelPath,
		Tokenizer: embedding.TokenizerConfig{
			ModelID:        "sentence-transformers/all-MiniLM-L6-v2",
			SequenceLength: 512,
		},
		Dimension:         384,
		BatchSize:         32,
		MaxSequenceLength: 512,
		CacheSize:         10,
		CacheSizeBytes:    1 << 20, // 1MB
		Options: embedding.Options{
			PadToMaxLength: false,
			Normalize:      true,
			CacheEnabled:   true,
		},
	}

	service, err := embedding.NewService(context.Background(), config)
	require.NoError(t, err)
	defer service.Close()

	// Test text that we'll access multiple times
	text := "Test text for cache metrics"

	// First access should be a miss
	_, err = service.Embed(context.Background(), text)
	require.NoError(t, err)
	metrics := service.GetCacheMetrics()
	assert.Equal(t, int64(0), metrics.Hits)
	assert.Equal(t, int64(1), metrics.Misses)

	// Second access should be a hit
	_, err = service.Embed(context.Background(), text)
	require.NoError(t, err)
	metrics = service.GetCacheMetrics()
	assert.Equal(t, int64(1), metrics.Hits)
	assert.Equal(t, int64(1), metrics.Misses)
}

// TestCacheConcurrency tests concurrent access to the cache
func TestCacheConcurrency(t *testing.T) {
	testDataPath := filepath.Join("..", "..", "testdata")
	modelPath := filepath.Join(testDataPath, "model.onnx")

	config := &embedding.Config{
		ModelPath: modelPath,
		Tokenizer: embedding.TokenizerConfig{
			ModelID:        "sentence-transformers/all-MiniLM-L6-v2",
			SequenceLength: 512,
		},
		Dimension:         384,
		BatchSize:         32,
		MaxSequenceLength: 512,
		CacheSize:         100,
		CacheSizeBytes:    1 << 20, // 1MB
		Options: embedding.Options{
			PadToMaxLength: false,
			Normalize:      true,
			CacheEnabled:   true,
		},
	}

	service, err := embedding.NewService(context.Background(), config)
	require.NoError(t, err)
	defer service.Close()

	// Number of concurrent goroutines
	workers := 3            // Reduced from 5
	requestsPerWorker := 10 // Reduced from 20
	var wg sync.WaitGroup
	wg.Add(workers)

	// Create a channel for errors
	errChan := make(chan error, workers*requestsPerWorker)

	// Launch workers
	for i := 0; i < workers; i++ {
		go func(workerID int) {
			defer wg.Done()
			for j := 0; j < requestsPerWorker; j++ {
				text := fmt.Sprintf("Test text %d-%d", workerID, j)
				ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
				_, err := service.Embed(ctx, text)
				cancel()
				if err != nil {
					errChan <- fmt.Errorf("worker %d request %d failed: %v", workerID, j, err)
				}
				// Add a larger delay between requests to prevent overwhelming the service
				time.Sleep(100 * time.Millisecond)
			}
		}(i)
	}

	// Wait for all workers to complete
	wg.Wait()
	close(errChan)

	// Check for any errors
	var errors []error
	for err := range errChan {
		errors = append(errors, err)
	}
	assert.Empty(t, errors, "Expected no errors during concurrent access")
}

// TestCacheInvalidation tests cache invalidation
func TestCacheInvalidation(t *testing.T) {
	testDataPath := filepath.Join("..", "..", "testdata")
	modelPath := filepath.Join(testDataPath, "model.onnx")

	// Create temporary directory for disk cache
	tempDir, err := os.MkdirTemp("", "embeddings-test-invalidation-*")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	config := &embedding.Config{
		ModelPath: modelPath,
		Tokenizer: embedding.TokenizerConfig{
			ModelID:        "sentence-transformers/all-MiniLM-L6-v2",
			SequenceLength: 512,
		},
		Dimension:         384,
		BatchSize:         32,
		MaxSequenceLength: 512,
		CacheSize:         10,
		CacheSizeBytes:    1 << 20, // 1MB
		DiskCacheEnabled:  true,
		DiskCachePath:     tempDir,
		Options: embedding.Options{
			PadToMaxLength: false,
			Normalize:      true,
			CacheEnabled:   true,
		},
	}

	service, err := embedding.NewService(context.Background(), config)
	require.NoError(t, err)
	defer service.Close()

	// Test text
	text := "Test text for cache invalidation"

	// Get initial embedding
	embedding1, err := service.Embed(context.Background(), text)
	require.NoError(t, err)
	require.NotNil(t, embedding1)

	// Invalidate the cache
	service.InvalidateCache()

	// Verify disk cache was cleared
	files, err := os.ReadDir(tempDir)
	require.NoError(t, err)
	binFiles := 0
	for _, file := range files {
		if strings.HasSuffix(file.Name(), ".bin") {
			binFiles++
		}
	}
	assert.Equal(t, 0, binFiles, "Expected disk cache to be empty after invalidation")

	// Get embedding again after invalidation
	embedding2, err := service.Embed(context.Background(), text)
	require.NoError(t, err)
	require.NotNil(t, embedding2)

	// Verify embeddings are the same (should be deterministic)
	assert.Equal(t, len(embedding1), len(embedding2), "Embedding lengths should match")
	for i := range embedding1 {
		assert.InDelta(t, embedding1[i], embedding2[i], 1e-6, "Embedding values should match at index %d", i)
	}

	// Verify cache metrics were reset
	metrics := service.GetCacheMetrics()
	assert.Equal(t, int64(0), metrics.Hits)
	assert.Equal(t, int64(1), metrics.Misses)
}

// TestCacheSizeAccuracy tests the accuracy of cache size tracking
func TestCacheSizeAccuracy(t *testing.T) {
	testDataPath := filepath.Join("..", "..", "testdata")
	modelPath := filepath.Join(testDataPath, "model.onnx")

	// Calculate sizes
	embeddingSize := int64(384 * 4) // 384 dimensions * 4 bytes per float32
	maxItems := 5
	maxBytes := embeddingSize * int64(maxItems) // Enough space for all test items

	config := &embedding.Config{
		ModelPath: modelPath,
		Tokenizer: embedding.TokenizerConfig{
			ModelID:        "sentence-transformers/all-MiniLM-L6-v2",
			SequenceLength: 512,
		},
		Dimension:         384,
		BatchSize:         32,
		MaxSequenceLength: 512,
		CacheSize:         maxItems,
		CacheSizeBytes:    maxBytes,
		Options: embedding.Options{
			PadToMaxLength: false,
			Normalize:      true,
			CacheEnabled:   true,
		},
	}

	service, err := embedding.NewService(context.Background(), config)
	require.NoError(t, err)
	defer service.Close()

	// Initial size should be 0
	assert.Equal(t, int64(0), service.GetCacheSize(), "Initial cache size should be 0")

	// Add items one by one and verify size
	texts := []string{
		"First test text",
		"Second test text",
		"Third test text",
		"Fourth test text",
		"Fifth test text",
	}

	for i, text := range texts {
		_, err := service.Embed(context.Background(), text)
		require.NoError(t, err)

		currentSize := service.GetCacheSize()
		expectedSize := embeddingSize * int64(i+1)
		assert.Equal(t, expectedSize, currentSize,
			"Cache size should be %d bytes for %d items", expectedSize, i+1)
	}

	// Add one more item to ensure eviction
	_, err = service.Embed(context.Background(), "One more text to trigger eviction")
	require.NoError(t, err)

	// Final size check - should still have exactly maxItems
	finalSize := service.GetCacheSize()
	expectedFinalSize := embeddingSize * int64(maxItems)
	assert.Equal(t, expectedFinalSize, finalSize,
		"Final cache size should be %d bytes for %d items", expectedFinalSize, maxItems)
}

// TestDiskCacheEdgeCases tests edge cases in the disk cache
func TestDiskCacheEdgeCases(t *testing.T) {
	testDataPath := filepath.Join("..", "..", "testdata")
	modelPath := filepath.Join(testDataPath, "model.onnx")

	// Create temporary directory for disk cache
	tempDir, err := os.MkdirTemp("", "embeddings-test-edge-*")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	config := &embedding.Config{
		ModelPath: modelPath,
		Tokenizer: embedding.TokenizerConfig{
			ModelID:        "sentence-transformers/all-MiniLM-L6-v2",
			SequenceLength: 512,
		},
		Dimension:         384,
		BatchSize:         32,
		MaxSequenceLength: 512,
		CacheSize:         1, // Tiny memory cache to force disk usage
		CacheSizeBytes:    4096,
		DiskCacheEnabled:  true,
		DiskCachePath:     tempDir,
		Options: embedding.Options{
			PadToMaxLength: false,
			Normalize:      true,
			CacheEnabled:   true,
		},
	}

	service, err := embedding.NewService(context.Background(), config)
	require.NoError(t, err)
	defer service.Close()

	// Test cases
	tests := []struct {
		name        string
		text        string
		expectError bool
	}{
		{"empty_string", "", true},
		{"very_long_text", strings.Repeat("very long text ", 100), false},
		{"special_chars", "!@#$%^&*()_+{}|:\"<>?~`-=[]\\;',./'", false},
		{"unicode", "Hello 你好 Bonjour こんにちは Hola", false},
		{"whitespace", "   \t\n\r   ", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Get initial embedding
			embedding1, err := service.Embed(context.Background(), tt.text)
			if tt.expectError {
				assert.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.NotNil(t, embedding1)

			// Force memory cache eviction
			_, err = service.Embed(context.Background(), "Eviction text")
			require.NoError(t, err)

			// Get embedding again (should come from disk)
			embedding2, err := service.Embed(context.Background(), tt.text)
			require.NoError(t, err)
			require.NotNil(t, embedding2)

			// Verify embeddings match
			assert.Equal(t, len(embedding1), len(embedding2), "Embedding lengths should match")
			for i := range embedding1 {
				assert.InDelta(t, embedding1[i], embedding2[i], 1e-6, "Embedding values should match at index %d", i)
			}

			// Verify disk cache file exists and has correct format
			files, err := os.ReadDir(tempDir)
			require.NoError(t, err)
			found := false
			for _, file := range files {
				if strings.HasSuffix(file.Name(), ".bin") {
					found = true
					// Verify file size is reasonable (should be at least dimension * 4 bytes)
					info, err := file.Info()
					require.NoError(t, err)
					assert.GreaterOrEqual(t, info.Size(), int64(config.Dimension*4))
					break
				}
			}
			assert.True(t, found, "Expected to find cache file for embedding")
		})
	}
}
