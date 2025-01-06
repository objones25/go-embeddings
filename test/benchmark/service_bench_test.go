package benchmark

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/objones25/go-embeddings/pkg/embedding"
)

func setupBenchmarkService(b *testing.B, enableCoreML bool) embedding.Service {
	// Get test data paths
	workspaceRoot := filepath.Join("..", "..")
	testDataPath := filepath.Join(workspaceRoot, "testdata")
	modelPath := filepath.Join(testDataPath, "model.onnx")
	tokenizerPath := filepath.Join(testDataPath, "tokenizer.json")

	// Set up library paths
	libPath := filepath.Join(workspaceRoot, "libs")
	os.Setenv("ONNXRUNTIME_LIB_PATH", filepath.Join(libPath, "libonnxruntime.1.20.0.dylib"))
	os.Setenv("DYLD_LIBRARY_PATH", libPath)
	os.Setenv("DYLD_FALLBACK_LIBRARY_PATH", libPath)
	os.Setenv("CGO_LDFLAGS", fmt.Sprintf("-L%s -ltokenizers -lonnxruntime -Wl,-rpath,%s", libPath, libPath))
	os.Setenv("CGO_CFLAGS", fmt.Sprintf("-I%s", filepath.Join(workspaceRoot, "onnxruntime-osx-arm64-1.20.0", "include")))

	config := &embedding.Config{
		ModelPath:   modelPath,
		EnableMetal: enableCoreML,
		CoreMLConfig: &embedding.CoreMLConfig{
			EnableCaching: true,
			RequireANE:    false,
		},
		BatchSize:         32,
		MaxSequenceLength: 512,
		Dimension:         384,
		Tokenizer: embedding.TokenizerConfig{
			LocalPath:      tokenizerPath,
			SequenceLength: 512,
		},
		Options: embedding.Options{
			Normalize:    true,
			CacheEnabled: true,
		},
	}

	service, err := embedding.NewService(context.Background(), config)
	if err != nil {
		b.Fatal(err)
	}
	return service
}

func BenchmarkEmbed(b *testing.B) {
	if runtime.GOOS != "darwin" {
		b.Skip("Skipping benchmark on non-Darwin platform")
	}

	tests := []struct {
		name        string
		enableMetal bool
	}{
		{"CPU", false},
		{"CoreML", true},
	}

	text := "This is a sample text for benchmarking the embedding service performance."
	ctx := context.Background()

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			service := setupBenchmarkService(b, tt.enableMetal)
			defer service.Close()

			// Warmup
			_, err := service.Embed(ctx, text)
			if err != nil {
				b.Fatal(err)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := service.Embed(ctx, text)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkBatchEmbed(b *testing.B) {
	if runtime.GOOS != "darwin" {
		b.Skip("Skipping benchmark on non-Darwin platform")
	}

	tests := []struct {
		name        string
		enableMetal bool
		batchSize   int
	}{
		{"CPU_Small_Batch", false, 4},
		{"CoreML_Small_Batch", true, 4},
		{"CPU_Medium_Batch", false, 16},
		{"CoreML_Medium_Batch", true, 16},
		{"CPU_Large_Batch", false, 32},
		{"CoreML_Large_Batch", true, 32},
	}

	texts := []string{
		"First sample text for batch processing.",
		"Second sample text with different content.",
		"Third sample text to test batch performance.",
		"Fourth sample text for comprehensive testing.",
	}

	ctx := context.Background()

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			service := setupBenchmarkService(b, tt.enableMetal)
			defer service.Close()

			// Create batch of required size
			batchTexts := make([]string, tt.batchSize)
			for i := 0; i < tt.batchSize; i++ {
				batchTexts[i] = texts[i%len(texts)]
			}

			// Warmup
			_, err := service.BatchEmbed(ctx, batchTexts)
			if err != nil {
				b.Fatal(err)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := service.BatchEmbed(ctx, batchTexts)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkBatchEmbedAsync(b *testing.B) {
	if runtime.GOOS != "darwin" {
		b.Skip("Skipping benchmark on non-Darwin platform")
	}

	tests := []struct {
		name        string
		enableMetal bool
		batchSize   int
	}{
		{"CPU_Async_Small", false, 4},
		{"CoreML_Async_Small", true, 4},
		{"CPU_Async_Large", false, 32},
		{"CoreML_Async_Large", true, 32},
	}

	texts := []string{
		"First sample text for async processing.",
		"Second sample text with different content.",
		"Third sample text to test async performance.",
		"Fourth sample text for comprehensive testing.",
	}

	ctx := context.Background()

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			service := setupBenchmarkService(b, tt.enableMetal)
			defer service.Close()

			// Create batch of required size
			batchTexts := make([]string, tt.batchSize)
			for i := 0; i < tt.batchSize; i++ {
				batchTexts[i] = texts[i%len(texts)]
			}

			// Warmup
			results := make(chan embedding.Result)
			errors := make(chan error)
			err := service.BatchEmbedAsync(ctx, batchTexts, results, errors)
			if err != nil {
				b.Fatal(err)
			}
			for range batchTexts {
				select {
				case result := <-results:
					if result.Error != nil {
						b.Fatal(result.Error)
					}
				case err := <-errors:
					b.Fatal(err)
				}
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				results := make(chan embedding.Result)
				errors := make(chan error)

				err := service.BatchEmbedAsync(ctx, batchTexts, results, errors)
				if err != nil {
					b.Fatal(err)
				}

				// Collect results
				for range batchTexts {
					select {
					case result := <-results:
						if result.Error != nil {
							b.Fatal(result.Error)
						}
					case err := <-errors:
						b.Fatal(err)
					case <-ctx.Done():
						b.Fatal(ctx.Err())
					}
				}
			}
		})
	}
}

func BenchmarkEmbedWithCache(b *testing.B) {
	if runtime.GOOS != "darwin" {
		b.Skip("Skipping benchmark on non-Darwin platform")
	}

	tests := []struct {
		name        string
		enableMetal bool
	}{
		{"CPU_Cached", false},
		{"CoreML_Cached", true},
	}

	text := "This is a sample text that should be cached after the first embedding."
	ctx := context.Background()

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			service := setupBenchmarkService(b, tt.enableMetal)
			defer service.Close()

			// Warm up cache
			_, err := service.Embed(ctx, text)
			if err != nil {
				b.Fatal(err)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := service.Embed(ctx, text)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
