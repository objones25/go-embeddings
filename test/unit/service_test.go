package unit

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/objones25/go-embeddings/pkg/embedding"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func init() {
	// Enable debug logging
	log.SetFlags(log.Lshortfile | log.LstdFlags)

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

	// Log important environment variables
	log.Printf("DYLD_LIBRARY_PATH=%s", os.Getenv("DYLD_LIBRARY_PATH"))
	log.Printf("LD_LIBRARY_PATH=%s", os.Getenv("LD_LIBRARY_PATH"))
	log.Printf("ONNXRUNTIME_LIB_PATH=%s", os.Getenv("ONNXRUNTIME_LIB_PATH"))
	log.Printf("CGO_LDFLAGS=%s", os.Getenv("CGO_LDFLAGS"))
	log.Printf("PWD=%s", os.Getenv("PWD"))

	// Log library files
	files, err := os.ReadDir(libsDir)
	if err != nil {
		log.Printf("Error reading libs dir: %v", err)
	} else {
		log.Printf("Library files in %s:", libsDir)
		for _, file := range files {
			info, err := file.Info()
			if err != nil {
				log.Printf("  %s (error getting info: %v)", file.Name(), err)
			} else {
				log.Printf("  %s (%d bytes)", file.Name(), info.Size())
			}
		}
	}

	// Create symlink for ONNX Runtime library
	onnxRuntimePath := filepath.Join(libsDir, "libonnxruntime.1.20.0.dylib")
	onnxRuntimeSymlink := filepath.Join(libsDir, "libonnxruntime.dylib")
	if err := os.Remove(onnxRuntimeSymlink); err != nil && !os.IsNotExist(err) {
		log.Printf("Error removing existing symlink: %v", err)
	}
	if err := os.Symlink(onnxRuntimePath, onnxRuntimeSymlink); err != nil {
		log.Printf("Error creating symlink: %v", err)
	}

	// Log test data paths
	testDataPath := filepath.Join(workspaceDir, "testdata")
	if _, err := os.Stat(testDataPath); os.IsNotExist(err) {
		log.Printf("Warning: Test data directory does not exist")
	}

	modelPath := filepath.Join(testDataPath, "model.onnx")
	tokenizerPath := filepath.Join(testDataPath, "tokenizer.json")

	log.Printf("Model path: %s (exists: %v)", modelPath, fileExists(modelPath))
	log.Printf("Tokenizer path: %s (exists: %v)", tokenizerPath, fileExists(tokenizerPath))
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
			log.Printf("Running test case: %s", tt.name)
			if tt.config != nil && tt.config.ModelPath != "" {
				log.Printf("Using model path: %s", tt.config.ModelPath)
			}

			service, err := embedding.NewService(context.Background(), tt.config)
			if tt.expectError {
				assert.Error(t, err)
				if err != nil {
					log.Printf("Got expected error: %v", err)
				}
				assert.Nil(t, service)
			} else {
				if err != nil {
					log.Printf("Unexpected error: %v", err)
				}
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

	log.Printf("Creating service with config: %+v", config)
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
			log.Printf("Running test case: %s with text: %q", tt.name, tt.text)
			embedding, err := service.Embed(tt.ctx, tt.text)
			if tt.expectError {
				assert.Error(t, err)
				if err != nil {
					log.Printf("Got expected error: %v", err)
				}
				assert.Nil(t, embedding)
			} else {
				if err != nil {
					log.Printf("Unexpected error: %v", err)
				}
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

	log.Printf("Creating service with config: %+v", config)
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
			log.Printf("Running test case: %s with %d texts", tt.name, len(tt.texts))
			embeddings, err := service.BatchEmbed(tt.ctx, tt.texts)
			if tt.expectError {
				assert.Error(t, err)
				if err != nil {
					log.Printf("Got expected error: %v", err)
				}
				assert.Nil(t, embeddings)
			} else {
				if err != nil {
					log.Printf("Unexpected error: %v", err)
				}
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

	log.Printf("Creating service with config: %+v", config)
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
			// Create a context with timeout
			ctx, cancel := context.WithTimeout(tt.ctx, 30*time.Second)
			defer cancel()

			log.Printf("Running test case: %s with %d texts", tt.name, len(tt.texts))
			results := make(chan embedding.Result)
			errors := make(chan error)

			log.Printf("Starting BatchEmbedAsync...")
			err := service.BatchEmbedAsync(ctx, tt.texts, results, errors)
			if tt.expectError {
				assert.Error(t, err)
				if err != nil {
					log.Printf("Got expected error: %v", err)
				}
				return
			}
			assert.NoError(t, err)

			// Collect results and errors
			var receivedResults []embedding.Result
			var receivedErrors []error
			done := make(chan struct{})

			log.Printf("Starting result collection goroutine...")
			go func() {
				defer close(done)
				expectedResults := len(tt.texts)
				receivedCount := 0

				for receivedCount < expectedResults {
					select {
					case result, ok := <-results:
						if !ok {
							log.Printf("Results channel closed")
							return
						}
						log.Printf("Received result for text: %q (embedding dimension: %d)", result.Text, len(result.Embedding))
						receivedResults = append(receivedResults, result)
						receivedCount++
					case err, ok := <-errors:
						if !ok {
							log.Printf("Errors channel closed")
							return
						}
						log.Printf("Received error: %v", err)
						receivedErrors = append(receivedErrors, err)
						return
					case <-ctx.Done():
						log.Printf("Context cancelled or timed out")
						return
					}
				}
			}()

			log.Printf("Waiting for done signal...")
			select {
			case <-done:
				log.Printf("Done signal received")
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
