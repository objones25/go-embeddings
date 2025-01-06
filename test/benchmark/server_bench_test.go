package benchmark

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/objones25/go-embeddings/pkg/embedding"
	"github.com/objones25/go-embeddings/pkg/server"
)

func setupBenchmarkServer(b *testing.B, enableCoreML bool) (*server.Server, func()) {
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
		b.Fatal(err)
	}

	cleanup := func() {
		srv.Shutdown(context.Background())
		service.Close()
	}

	return srv, cleanup
}

func BenchmarkHTTPEmbed(b *testing.B) {
	if runtime.GOOS != "darwin" {
		b.Skip("Skipping benchmark on non-Darwin platform")
	}

	tests := []struct {
		name        string
		enableMetal bool
		textLength  int
	}{
		{"CPU_Short_Text", false, 20},
		{"CoreML_Short_Text", true, 20},
		{"CPU_Medium_Text", false, 100},
		{"CoreML_Medium_Text", true, 100},
		{"CPU_Long_Text", false, 500},
		{"CoreML_Long_Text", true, 500},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			srv, cleanup := setupBenchmarkServer(b, tt.enableMetal)
			defer cleanup()

			text := strings.Repeat("This is a test sentence. ", tt.textLength/20)
			requestBody := map[string]interface{}{
				"text": text,
			}
			body, err := json.Marshal(requestBody)
			if err != nil {
				b.Fatal(err)
			}

			// Warmup
			req := httptest.NewRequest("POST", "/api/v1/embed", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()
			srv.GetRouter().ServeHTTP(w, req)
			if w.Code != http.StatusOK {
				b.Fatalf("warmup failed: expected status OK, got %v", w.Code)
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
	if runtime.GOOS != "darwin" {
		b.Skip("Skipping benchmark on non-Darwin platform")
	}

	tests := []struct {
		name        string
		enableMetal bool
		batchSize   int
		textLength  int
	}{
		{"CPU_Small_Batch_Short", false, 4, 20},
		{"CoreML_Small_Batch_Short", true, 4, 20},
		{"CPU_Medium_Batch_Medium", false, 16, 100},
		{"CoreML_Medium_Batch_Medium", true, 16, 100},
		{"CPU_Large_Batch_Long", false, 32, 200},
		{"CoreML_Large_Batch_Long", true, 32, 200},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			srv, cleanup := setupBenchmarkServer(b, tt.enableMetal)
			defer cleanup()

			// Create batch of texts
			texts := make([]string, tt.batchSize)
			for i := range texts {
				texts[i] = strings.Repeat("This is a test sentence. ", tt.textLength/20)
			}

			requestBody := map[string]interface{}{
				"texts": texts,
			}
			body, err := json.Marshal(requestBody)
			if err != nil {
				b.Fatal(err)
			}

			// Warmup
			req := httptest.NewRequest("POST", "/api/v1/embed/batch", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()
			srv.GetRouter().ServeHTTP(w, req)
			if w.Code != http.StatusOK {
				b.Fatalf("warmup failed: expected status OK, got %v", w.Code)
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
	if runtime.GOOS != "darwin" {
		b.Skip("Skipping benchmark on non-Darwin platform")
	}

	tests := []struct {
		name        string
		enableMetal bool
	}{
		{"CPU_Parallel", false},
		{"CoreML_Parallel", true},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			srv, cleanup := setupBenchmarkServer(b, tt.enableMetal)
			defer cleanup()

			requestBody := map[string]interface{}{
				"text": "This is a sample text for benchmarking parallel HTTP server performance.",
			}
			body, err := json.Marshal(requestBody)
			if err != nil {
				b.Fatal(err)
			}

			// Warmup
			req := httptest.NewRequest("POST", "/api/v1/embed", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()
			srv.GetRouter().ServeHTTP(w, req)
			if w.Code != http.StatusOK {
				b.Fatalf("warmup failed: expected status OK, got %v", w.Code)
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
		})
	}
}

func BenchmarkHTTPBatchEmbedParallel(b *testing.B) {
	if runtime.GOOS != "darwin" {
		b.Skip("Skipping benchmark on non-Darwin platform")
	}

	tests := []struct {
		name        string
		enableMetal bool
		batchSize   int
	}{
		{"CPU_Parallel_Small_Batch", false, 4},
		{"CoreML_Parallel_Small_Batch", true, 4},
		{"CPU_Parallel_Large_Batch", false, 32},
		{"CoreML_Parallel_Large_Batch", true, 32},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			srv, cleanup := setupBenchmarkServer(b, tt.enableMetal)
			defer cleanup()

			// Create batch of texts
			texts := make([]string, tt.batchSize)
			for i := range texts {
				texts[i] = fmt.Sprintf("Sample text %d for parallel batch processing.", i)
			}

			requestBody := map[string]interface{}{
				"texts": texts,
			}
			body, err := json.Marshal(requestBody)
			if err != nil {
				b.Fatal(err)
			}

			// Warmup
			req := httptest.NewRequest("POST", "/api/v1/embed/batch", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()
			srv.GetRouter().ServeHTTP(w, req)
			if w.Code != http.StatusOK {
				b.Fatalf("warmup failed: expected status OK, got %v", w.Code)
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
		})
	}
}

func BenchmarkMemoryHTTPEmbed(b *testing.B) {
	if runtime.GOOS != "darwin" {
		b.Skip("Skipping benchmark on non-Darwin platform")
	}

	tests := []struct {
		name        string
		enableMetal bool
	}{
		{"CPU_Memory", false},
		{"CoreML_Memory", true},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			srv, cleanup := setupBenchmarkServer(b, tt.enableMetal)
			defer cleanup()

			requestBody := map[string]interface{}{
				"text": strings.Repeat("Memory benchmark test text. ", 20),
			}
			body, err := json.Marshal(requestBody)
			if err != nil {
				b.Fatal(err)
			}

			// Warmup
			req := httptest.NewRequest("POST", "/api/v1/embed", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()
			srv.GetRouter().ServeHTTP(w, req)
			if w.Code != http.StatusOK {
				b.Fatalf("warmup failed: expected status OK, got %v", w.Code)
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
		})
	}
}
