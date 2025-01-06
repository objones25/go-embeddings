package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/objones25/go-embeddings/pkg/embedding"
	"github.com/objones25/go-embeddings/pkg/server"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func init() {
	// Enable debug logging
	log.SetFlags(log.Lshortfile | log.LstdFlags)

	// Get absolute paths
	workspaceDir, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		log.Printf("Error getting workspace dir: %v", err)
	}
	libsDir := filepath.Join(workspaceDir, "libs")
	onnxRuntimePath := filepath.Join(libsDir, "libonnxruntime.1.20.0.dylib")

	// Set environment variables with absolute paths
	os.Setenv("DYLD_LIBRARY_PATH", libsDir)
	os.Setenv("LD_LIBRARY_PATH", libsDir)
	os.Setenv("ONNXRUNTIME_LIB_PATH", onnxRuntimePath)
	os.Setenv("CGO_LDFLAGS", fmt.Sprintf("-L%s", libsDir))
	os.Setenv("CGO_CFLAGS", fmt.Sprintf("-I%s", filepath.Join(workspaceDir, "onnxruntime-osx-arm64-1.20.0", "include")))

	// Create symlinks for ONNX Runtime library
	onnxRuntimeSymlink := filepath.Join(libsDir, "libonnxruntime.dylib")
	if err := os.Remove(onnxRuntimeSymlink); err != nil && !os.IsNotExist(err) {
		log.Printf("Error removing existing symlink: %v", err)
	}
	if err := os.Symlink(onnxRuntimePath, onnxRuntimeSymlink); err != nil {
		log.Printf("Error creating symlink: %v", err)
	}

	// Create additional symlink for .so extension
	onnxRuntimeSoSymlink := filepath.Join(libsDir, "onnxruntime.so")
	if err := os.Remove(onnxRuntimeSoSymlink); err != nil && !os.IsNotExist(err) {
		log.Printf("Error removing existing .so symlink: %v", err)
	}
	if err := os.Symlink(onnxRuntimePath, onnxRuntimeSoSymlink); err != nil {
		log.Printf("Error creating .so symlink: %v", err)
	}

	// Log environment setup
	log.Printf("Integration test environment setup:")
	log.Printf("Workspace dir: %s", workspaceDir)
	log.Printf("DYLD_LIBRARY_PATH=%s", os.Getenv("DYLD_LIBRARY_PATH"))
	log.Printf("LD_LIBRARY_PATH=%s", os.Getenv("LD_LIBRARY_PATH"))
	log.Printf("ONNXRUNTIME_LIB_PATH=%s", os.Getenv("ONNXRUNTIME_LIB_PATH"))
	log.Printf("CGO_LDFLAGS=%s", os.Getenv("CGO_LDFLAGS"))
	log.Printf("CGO_CFLAGS=%s", os.Getenv("CGO_CFLAGS"))
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
}

// getAvailablePort returns an available port for testing
func getAvailablePort() (string, error) {
	addr, err := net.Listen("tcp", ":0")
	if err != nil {
		return "", err
	}
	defer addr.Close()

	// Get the dynamic port
	port := addr.Addr().(*net.TCPAddr).Port
	return strconv.Itoa(port), nil
}

func setupTestServer(t *testing.T) (*server.Server, func()) {
	// Get absolute paths
	workspaceDir, err := filepath.Abs(filepath.Join("..", ".."))
	require.NoError(t, err)

	testDataPath := filepath.Join(workspaceDir, "testdata")
	modelPath := filepath.Join(testDataPath, "model.onnx")
	libsDir := filepath.Join(workspaceDir, "libs")

	// Ensure ONNX runtime library path is set
	onnxPath := filepath.Join(libsDir, "libonnxruntime.1.20.0.dylib")
	os.Setenv("ONNXRUNTIME_LIB_PATH", onnxPath)

	// Get dynamic port for metrics
	metricsPort, err := getAvailablePort()
	require.NoError(t, err)

	config := &embedding.Config{
		ModelPath: modelPath,
		Tokenizer: embedding.TokenizerConfig{
			LocalPath:      filepath.Join(testDataPath, "tokenizer.json"),
			SequenceLength: 512,
		},
		BatchSize:         32,
		MaxSequenceLength: 512,
		Dimension:         384,
		Options: embedding.Options{
			Normalize:    true,
			CacheEnabled: true,
		},
	}

	t.Logf("Creating service with config: %+v", config)
	service, err := embedding.NewService(context.Background(), config)
	require.NoError(t, err)

	serverConfig := &server.Config{
		Port:           "8080",
		MetricsPort:    metricsPort,
		ReadTimeout:    10 * time.Second,
		WriteTimeout:   10 * time.Second,
		MaxRequestSize: 1 << 20, // 1MB
		AllowedOrigins: []string{"*"},
	}

	srv, err := server.NewServer(serverConfig, service)
	require.NoError(t, err)

	// Create cleanup function
	cleanup := func() {
		t.Log("Cleaning up test server")
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		if err := srv.Shutdown(ctx); err != nil {
			t.Logf("Error during server shutdown: %v", err)
		}
		service.Close()
	}

	return srv, cleanup
}

func TestEmbedEndpoint(t *testing.T) {
	srv, cleanup := setupTestServer(t)
	defer cleanup()

	tests := []struct {
		name         string
		requestBody  map[string]interface{}
		expectStatus int
		expectError  bool
		errorMessage string
	}{
		{
			name: "valid request",
			requestBody: map[string]interface{}{
				"text": "Hello, world!",
			},
			expectStatus: http.StatusOK,
			expectError:  false,
		},
		{
			name: "empty text",
			requestBody: map[string]interface{}{
				"text": "",
			},
			expectStatus: http.StatusBadRequest,
			expectError:  true,
			errorMessage: "empty text",
		},
		{
			name:         "missing text",
			requestBody:  map[string]interface{}{},
			expectStatus: http.StatusBadRequest,
			expectError:  true,
			errorMessage: "missing text field",
		},
		{
			name: "text too long",
			requestBody: map[string]interface{}{
				"text": strings.Repeat("a", 10000),
			},
			expectStatus: http.StatusOK,
			expectError:  false,
		},
		{
			name: "invalid json",
			requestBody: map[string]interface{}{
				"text": make(chan int),
			},
			expectStatus: http.StatusBadRequest,
			expectError:  true,
			errorMessage: "invalid request body",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var body []byte
			var err error

			if _, ok := tt.requestBody["text"].(chan int); ok {
				body = []byte("{invalid json}")
			} else {
				body, err = json.Marshal(tt.requestBody)
				require.NoError(t, err)
			}

			req := httptest.NewRequest("POST", "/api/v1/embed", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			srv.GetRouter().ServeHTTP(w, req)

			assert.Equal(t, tt.expectStatus, w.Code)

			var response map[string]interface{}
			err = json.Unmarshal(w.Body.Bytes(), &response)
			require.NoError(t, err)

			if tt.expectError {
				assert.Contains(t, response, "error")
				if tt.errorMessage != "" {
					errorMsg, ok := response["error"].(string)
					assert.True(t, ok)
					assert.Contains(t, strings.ToLower(errorMsg), tt.errorMessage)
				}
			} else {
				assert.Contains(t, response, "embedding")
				embedding, ok := response["embedding"].([]interface{})
				assert.True(t, ok)
				assert.NotEmpty(t, embedding)
				assert.Equal(t, 384, len(embedding)) // Check correct dimension
			}
		})
	}
}

func TestBatchEmbedEndpoint(t *testing.T) {
	srv, cleanup := setupTestServer(t)
	defer cleanup()

	tests := []struct {
		name         string
		requestBody  map[string]interface{}
		expectStatus int
		expectError  bool
	}{
		{
			name: "valid batch",
			requestBody: map[string]interface{}{
				"texts": []string{
					"Hello, world!",
					"This is a test.",
				},
			},
			expectStatus: http.StatusOK,
			expectError:  false,
		},
		{
			name: "empty batch",
			requestBody: map[string]interface{}{
				"texts": []string{},
			},
			expectStatus: http.StatusInternalServerError,
			expectError:  true,
		},
		{
			name:         "missing texts",
			requestBody:  map[string]interface{}{},
			expectStatus: http.StatusBadRequest,
			expectError:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body, err := json.Marshal(tt.requestBody)
			require.NoError(t, err)

			req := httptest.NewRequest("POST", "/api/v1/embed/batch", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			srv.GetRouter().ServeHTTP(w, req)

			assert.Equal(t, tt.expectStatus, w.Code)

			var response map[string]interface{}
			err = json.Unmarshal(w.Body.Bytes(), &response)
			require.NoError(t, err)

			if tt.expectError {
				assert.Contains(t, response, "error")
			} else {
				assert.Contains(t, response, "embeddings")
				embeddings, ok := response["embeddings"].([]interface{})
				assert.True(t, ok)
				assert.Len(t, embeddings, len(tt.requestBody["texts"].([]string)))
			}
		})
	}
}

func TestAsyncEmbedEndpoint(t *testing.T) {
	srv, cleanup := setupTestServer(t)
	defer cleanup()

	tests := []struct {
		name         string
		requestBody  map[string]interface{}
		expectStatus int
		expectError  bool
	}{
		{
			name: "valid batch",
			requestBody: map[string]interface{}{
				"texts": []string{
					"Hello, world!",
					"This is a test.",
				},
			},
			expectStatus: http.StatusOK,
			expectError:  false,
		},
		{
			name: "empty batch",
			requestBody: map[string]interface{}{
				"texts": []string{},
			},
			expectStatus: http.StatusInternalServerError,
			expectError:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body, err := json.Marshal(tt.requestBody)
			require.NoError(t, err)

			req := httptest.NewRequest("POST", "/api/v1/embed/async", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			srv.GetRouter().ServeHTTP(w, req)

			assert.Equal(t, tt.expectStatus, w.Code)

			if !tt.expectError {
				// For SSE responses, we should receive multiple events
				events := bytes.Split(w.Body.Bytes(), []byte("\n\n"))
				assert.NotEmpty(t, events)

				for _, event := range events {
					if len(event) == 0 {
						continue
					}
					var result embedding.Result
					err = json.Unmarshal(event, &result)
					if err != nil {
						// Skip error events
						continue
					}
					assert.NotEmpty(t, result.Embedding)
				}
			}
		})
	}
}

func TestHealthEndpoint(t *testing.T) {
	srv, cleanup := setupTestServer(t)
	defer cleanup()

	req := httptest.NewRequest("GET", "/health", nil)
	w := httptest.NewRecorder()

	srv.GetRouter().ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var response map[string]string
	err := json.Unmarshal(w.Body.Bytes(), &response)
	require.NoError(t, err)
	assert.Equal(t, "ok", response["status"])
}
