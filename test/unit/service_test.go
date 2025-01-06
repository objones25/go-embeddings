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
