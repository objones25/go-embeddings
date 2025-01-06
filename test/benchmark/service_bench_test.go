package benchmark

import (
	"context"
	"testing"

	"github.com/objones25/go-embeddings/pkg/embedding"
)

func setupBenchmarkService() (embedding.Service, error) {
	config := &embedding.Config{
		ModelPath: "../../testdata/model.onnx",
		Tokenizer: embedding.TokenizerConfig{
			ModelID: "bert-base-uncased",
		},
		BatchSize:         32,
		MaxSequenceLength: 512,
		Dimension:         768,
		Options: embedding.Options{
			Normalize:    true,
			CacheEnabled: true,
		},
	}

	return embedding.NewService(context.Background(), config)
}

func BenchmarkEmbed(b *testing.B) {
	service, err := setupBenchmarkService()
	if err != nil {
		b.Fatal(err)
	}
	defer service.Close()

	text := "This is a sample text for benchmarking the embedding service performance."
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := service.Embed(ctx, text)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkBatchEmbed(b *testing.B) {
	service, err := setupBenchmarkService()
	if err != nil {
		b.Fatal(err)
	}
	defer service.Close()

	texts := []string{
		"First sample text for batch processing.",
		"Second sample text with different content.",
		"Third sample text to test batch performance.",
		"Fourth sample text for comprehensive testing.",
	}
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := service.BatchEmbed(ctx, texts)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkBatchEmbedParallel(b *testing.B) {
	service, err := setupBenchmarkService()
	if err != nil {
		b.Fatal(err)
	}
	defer service.Close()

	texts := []string{
		"First sample text for batch processing.",
		"Second sample text with different content.",
		"Third sample text to test batch performance.",
		"Fourth sample text for comprehensive testing.",
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		ctx := context.Background()
		for pb.Next() {
			_, err := service.BatchEmbed(ctx, texts)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkEmbedWithCache(b *testing.B) {
	service, err := setupBenchmarkService()
	if err != nil {
		b.Fatal(err)
	}
	defer service.Close()

	text := "This is a sample text that should be cached after the first embedding."
	ctx := context.Background()

	// Warm up cache
	_, err = service.Embed(ctx, text)
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
}

func BenchmarkBatchEmbedAsync(b *testing.B) {
	service, err := setupBenchmarkService()
	if err != nil {
		b.Fatal(err)
	}
	defer service.Close()

	texts := []string{
		"First sample text for async processing.",
		"Second sample text with different content.",
		"Third sample text to test async performance.",
		"Fourth sample text for comprehensive testing.",
	}
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		results := make(chan embedding.Result)
		errors := make(chan error)

		err := service.BatchEmbedAsync(ctx, texts, results, errors)
		if err != nil {
			b.Fatal(err)
		}

		// Collect results
		for range texts {
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
}
