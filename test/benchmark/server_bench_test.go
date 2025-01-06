package benchmark

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/objones25/go-embeddings/pkg/embedding"
	"github.com/objones25/go-embeddings/pkg/server"
)

func setupBenchmarkServer() (*server.Server, func()) {
	config := &embedding.Config{
		ModelPath: "../../testdata/model.onnx",
		Tokenizer: embedding.TokenizerConfig{
			LocalPath: "../../testdata/tokenizer.json",
		},
		BatchSize:         32,
		MaxSequenceLength: 512,
		Dimension:         384,
		Options: embedding.Options{
			Normalize:    true,
			CacheEnabled: true,
		},
	}

	service, err := embedding.NewService(context.Background(), config)
	if err != nil {
		panic(err)
	}

	serverConfig := &server.Config{
		Port:           "8080",
		MetricsPort:    "8081",
		ReadTimeout:    10 * time.Second,
		WriteTimeout:   10 * time.Second,
		MaxRequestSize: 1 << 20, // 1MB
		AllowedOrigins: []string{"*"},
	}

	srv, err := server.NewServer(serverConfig, service)
	if err != nil {
		service.Close()
		panic(err)
	}

	return srv, func() {
		srv.Shutdown(context.Background())
		service.Close()
	}
}

func BenchmarkHTTPEmbed(b *testing.B) {
	srv, cleanup := setupBenchmarkServer()
	defer cleanup()

	scenarios := []struct {
		name string
		text string
	}{
		{
			name: "Short_Text",
			text: "This is a short text.",
		},
		{
			name: "Medium_Text",
			text: "This is a medium length text that contains multiple words and should test the performance with a more realistic input size.",
		},
		{
			name: "Long_Text",
			text: strings.Repeat("This is a very long text with repeated content to test performance with large inputs. ", 10),
		},
	}

	for _, scenario := range scenarios {
		b.Run(scenario.name, func(b *testing.B) {
			requestBody := map[string]interface{}{
				"text": scenario.text,
			}
			body, err := json.Marshal(requestBody)
			if err != nil {
				b.Fatal(err)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				req := httptest.NewRequest("POST", "/api/v1/embed", bytes.NewReader(body))
				req.Header.Set("Content-Type", "application/json")
				w := httptest.NewRecorder()

				srv.GetRouter().ServeHTTP(w, req)

				if w.Code != http.StatusOK {
					b.Fatalf("expected status OK, got %v", w.Code)
				}
			}
		})
	}
}

func BenchmarkHTTPBatchEmbed(b *testing.B) {
	srv, cleanup := setupBenchmarkServer()
	defer cleanup()

	scenarios := []struct {
		name       string
		batchSize  int
		textLength int
	}{
		{
			name:       "Small_Batch_Short_Texts",
			batchSize:  2,
			textLength: 10,
		},
		{
			name:       "Medium_Batch_Medium_Texts",
			batchSize:  8,
			textLength: 50,
		},
		{
			name:       "Large_Batch_Long_Texts",
			batchSize:  32,
			textLength: 100,
		},
	}

	for _, scenario := range scenarios {
		b.Run(scenario.name, func(b *testing.B) {
			texts := make([]string, scenario.batchSize)
			for i := range texts {
				texts[i] = strings.Repeat("a", scenario.textLength)
			}

			requestBody := map[string]interface{}{
				"texts": texts,
			}
			body, err := json.Marshal(requestBody)
			if err != nil {
				b.Fatal(err)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				req := httptest.NewRequest("POST", "/api/v1/embed/batch", bytes.NewReader(body))
				req.Header.Set("Content-Type", "application/json")
				w := httptest.NewRecorder()

				srv.GetRouter().ServeHTTP(w, req)

				if w.Code != http.StatusOK {
					b.Fatalf("expected status OK, got %v", w.Code)
				}
			}
		})
	}
}

func BenchmarkHTTPEmbedParallel(b *testing.B) {
	srv, cleanup := setupBenchmarkServer()
	defer cleanup()

	requestBody := map[string]interface{}{
		"text": "This is a sample text for benchmarking parallel HTTP server performance.",
	}
	body, err := json.Marshal(requestBody)
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			req := httptest.NewRequest("POST", "/api/v1/embed", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			srv.GetRouter().ServeHTTP(w, req)

			if w.Code != http.StatusOK {
				b.Fatalf("expected status OK, got %v", w.Code)
			}
		}
	})
}

func BenchmarkHTTPBatchEmbedParallel(b *testing.B) {
	srv, cleanup := setupBenchmarkServer()
	defer cleanup()

	requestBody := map[string]interface{}{
		"texts": []string{
			"First sample text for parallel batch processing.",
			"Second sample text with different content.",
			"Third sample text to test parallel performance.",
			"Fourth sample text for comprehensive testing.",
		},
	}
	body, err := json.Marshal(requestBody)
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			req := httptest.NewRequest("POST", "/api/v1/embed/batch", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			srv.GetRouter().ServeHTTP(w, req)

			if w.Code != http.StatusOK {
				b.Fatalf("expected status OK, got %v", w.Code)
			}
		}
	})
}

func BenchmarkMemoryHTTPEmbed(b *testing.B) {
	srv, cleanup := setupBenchmarkServer()
	defer cleanup()

	requestBody := map[string]interface{}{
		"text": strings.Repeat("Memory benchmark test text. ", 20),
	}
	body, err := json.Marshal(requestBody)
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest("POST", "/api/v1/embed", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		srv.GetRouter().ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			b.Fatalf("expected status OK, got %v", w.Code)
		}
	}
}
