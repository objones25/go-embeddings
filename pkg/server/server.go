package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/joho/godotenv"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.uber.org/zap"

	"github.com/objones25/go-embeddings/pkg/embedding"
)

// Server represents the HTTP server
type Server struct {
	config        *Config
	service       embedding.Service
	router        *chi.Mux
	logger        *zap.Logger
	metrics       *Metrics
	metricsServer *http.Server
}

// Config holds server configuration
type Config struct {
	Port              string
	MetricsPort       string
	ReadTimeout       time.Duration
	WriteTimeout      time.Duration
	MaxRequestSize    int64
	AllowedOrigins    []string
	MaxSequenceLength int `env:"MAX_SEQUENCE_LENGTH" envDefault:"512"`
}

// NewConfig creates a new server configuration from environment variables
func NewConfig() (*Config, error) {
	if err := godotenv.Load(); err != nil {
		// It's okay if .env doesn't exist
		if !os.IsNotExist(err) {
			return nil, fmt.Errorf("error loading .env file: %v", err)
		}
	}

	maxSeqLen := os.Getenv("MAX_SEQUENCE_LENGTH")
	if maxSeqLen == "" {
		maxSeqLen = "512" // Default value
	}

	maxSeqLenInt, err := strconv.Atoi(maxSeqLen)
	if err != nil {
		return nil, fmt.Errorf("invalid MAX_SEQUENCE_LENGTH: %v", err)
	}

	return &Config{
		Port:              getEnvWithDefault("PORT", "8080"),
		MetricsPort:       getEnvWithDefault("METRICS_PORT", "8081"),
		ReadTimeout:       time.Duration(getEnvAsIntWithDefault("READ_TIMEOUT_SECONDS", 30)) * time.Second,
		WriteTimeout:      time.Duration(getEnvAsIntWithDefault("WRITE_TIMEOUT_SECONDS", 30)) * time.Second,
		MaxRequestSize:    int64(getEnvAsIntWithDefault("MAX_REQUEST_SIZE_BYTES", 1024*1024)), // 1MB default
		AllowedOrigins:    strings.Split(getEnvWithDefault("ALLOWED_ORIGINS", "*"), ","),
		MaxSequenceLength: maxSeqLenInt,
	}, nil
}

// getEnvWithDefault gets an environment variable with a default value
func getEnvWithDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// getEnvAsIntWithDefault gets an environment variable as int with a default value
func getEnvAsIntWithDefault(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

// NewServer creates a new HTTP server
func NewServer(config *Config, service embedding.Service) (*Server, error) {
	logger, err := zap.NewProduction()
	if err != nil {
		return nil, err
	}

	// Create a new registry for metrics
	registry := prometheus.NewRegistry()
	metrics := NewMetrics()

	// Try to register metrics, ignoring already registered errors
	for _, collector := range []prometheus.Collector{
		metrics.requestDuration,
		metrics.requestsTotal,
		metrics.errorsTotal,
	} {
		if err := registry.Register(collector); err != nil {
			if _, ok := err.(prometheus.AlreadyRegisteredError); !ok {
				return nil, fmt.Errorf("failed to register metric: %v", err)
			}
		}
	}

	s := &Server{
		config:  config,
		service: service,
		router:  chi.NewRouter(),
		logger:  logger,
		metrics: metrics,
	}

	s.setupRoutes()
	return s, nil
}

// setupRoutes configures the server routes
func (s *Server) setupRoutes() {
	// Middleware
	s.router.Use(middleware.RequestID)
	s.router.Use(middleware.RealIP)
	s.router.Use(middleware.Logger)
	s.router.Use(middleware.Recoverer)
	s.router.Use(middleware.Timeout(60 * time.Second))
	s.router.Use(s.metricsMiddleware)
	s.router.Use(s.corsMiddleware)

	// API routes
	s.router.Route("/api/v1", func(r chi.Router) {
		r.Post("/embed", s.handleEmbed())
		r.Post("/embed/batch", s.handleBatchEmbed())
		r.Post("/embed/async", s.handleAsyncEmbed())
	})

	// Health check
	s.router.Get("/health", s.handleHealth())

	// Metrics endpoint on separate port
	metricsRouter := chi.NewRouter()
	metricsRouter.Handle("/metrics", promhttp.Handler())

	s.metricsServer = &http.Server{
		Addr:    ":" + s.config.MetricsPort,
		Handler: metricsRouter,
	}

	go func() {
		if err := s.metricsServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			s.logger.Error("metrics server failed", zap.Error(err))
		}
	}()
}

// Start starts the HTTP server
func (s *Server) Start() error {
	server := &http.Server{
		Addr:           ":" + s.config.Port,
		Handler:        s.router,
		ReadTimeout:    s.config.ReadTimeout,
		WriteTimeout:   s.config.WriteTimeout,
		MaxHeaderBytes: int(s.config.MaxRequestSize),
	}

	s.logger.Info("starting server", zap.String("port", s.config.Port))
	return server.ListenAndServe()
}

// handleEmbed handles single text embedding requests
func (s *Server) handleEmbed() http.HandlerFunc {
	type request struct {
		Text string `json:"text"`
	}

	type response struct {
		Embedding []float32 `json:"embedding"`
	}

	return func(w http.ResponseWriter, r *http.Request) {
		var req map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			s.handleError(w, "invalid request body", http.StatusBadRequest)
			return
		}

		text, ok := req["text"]
		if !ok {
			s.handleError(w, "missing text field", http.StatusBadRequest)
			return
		}

		textStr, ok := text.(string)
		if !ok {
			s.handleError(w, "invalid text field", http.StatusBadRequest)
			return
		}

		if textStr == "" {
			s.handleError(w, "empty text", http.StatusBadRequest)
			return
		}

		s.logger.Debug("processing embed request",
			zap.String("text", textStr))

		embedding, err := s.service.Embed(r.Context(), textStr)
		if err != nil {
			s.logger.Error("embedding failed",
				zap.Error(err),
				zap.String("text", textStr))
			s.handleError(w, err.Error(), http.StatusInternalServerError)
			return
		}

		s.writeJSON(w, response{Embedding: embedding})
	}
}

// handleBatchEmbed handles batch embedding requests
func (s *Server) handleBatchEmbed() http.HandlerFunc {
	type request struct {
		Texts []string `json:"texts"`
	}

	type response struct {
		Embeddings [][]float32 `json:"embeddings"`
	}

	return func(w http.ResponseWriter, r *http.Request) {
		var req map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			s.handleError(w, "invalid request body", http.StatusBadRequest)
			return
		}

		texts, ok := req["texts"]
		if !ok {
			s.handleError(w, "missing texts field", http.StatusBadRequest)
			return
		}

		textsArr, ok := texts.([]interface{})
		if !ok {
			s.handleError(w, "invalid texts field", http.StatusBadRequest)
			return
		}

		if len(textsArr) == 0 {
			s.handleError(w, "empty batch", http.StatusInternalServerError)
			return
		}

		// Convert interface array to string array
		textStrs := make([]string, len(textsArr))
		for i, text := range textsArr {
			textStr, ok := text.(string)
			if !ok {
				s.handleError(w, "invalid text in batch", http.StatusBadRequest)
				return
			}
			if textStr == "" {
				s.handleError(w, "empty text in batch", http.StatusBadRequest)
				return
			}
			textStrs[i] = textStr
		}

		s.logger.Debug("processing batch embed request",
			zap.Int("batch_size", len(textStrs)))

		embeddings, err := s.service.BatchEmbed(r.Context(), textStrs)
		if err != nil {
			s.handleError(w, err.Error(), http.StatusInternalServerError)
			return
		}

		s.writeJSON(w, response{Embeddings: embeddings})
	}
}

// handleAsyncEmbed handles asynchronous embedding requests
func (s *Server) handleAsyncEmbed() http.HandlerFunc {
	type request struct {
		Texts []string `json:"texts"`
	}

	return func(w http.ResponseWriter, r *http.Request) {
		var req request
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			s.handleError(w, "invalid JSON request", http.StatusBadRequest)
			return
		}

		if len(req.Texts) == 0 {
			s.handleError(w, "empty batch", http.StatusInternalServerError)
			return
		}

		// Create SSE response
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.WriteHeader(http.StatusOK)

		flusher, ok := w.(http.Flusher)
		if !ok {
			s.handleError(w, "streaming not supported", http.StatusInternalServerError)
			return
		}

		results := make(chan embedding.Result)
		errors := make(chan error)

		// Start async embedding
		if err := s.service.BatchEmbedAsync(r.Context(), req.Texts, results, errors); err != nil {
			s.handleError(w, err.Error(), http.StatusInternalServerError)
			return
		}

		// Stream results
		for {
			select {
			case result, ok := <-results:
				if !ok {
					return
				}
				if err := json.NewEncoder(w).Encode(result); err != nil {
					s.logger.Error("failed to encode result", zap.Error(err))
					return
				}
				flusher.Flush()
			case err, ok := <-errors:
				if !ok {
					return
				}
				s.logger.Error("async embedding error", zap.Error(err))
				if err := json.NewEncoder(w).Encode(map[string]string{"error": err.Error()}); err != nil {
					s.logger.Error("failed to encode error", zap.Error(err))
				}
				flusher.Flush()
			case <-r.Context().Done():
				return
			}
		}
	}
}

// handleHealth handles health check requests
func (s *Server) handleHealth() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		s.writeJSON(w, map[string]string{"status": "ok"})
	}
}

// metricsMiddleware collects metrics for each request
func (s *Server) metricsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		ww := middleware.NewWrapResponseWriter(w, r.ProtoMajor)

		defer func() {
			status := strconv.Itoa(ww.Status())
			s.metrics.requestDuration.WithLabelValues(r.Method, r.URL.Path, status).Observe(time.Since(start).Seconds())
			s.metrics.requestsTotal.WithLabelValues(r.Method, r.URL.Path).Inc()
			if ww.Status() >= 400 {
				s.metrics.errorsTotal.WithLabelValues(r.Method, r.URL.Path, status).Inc()
			}
		}()

		next.ServeHTTP(ww, r)
	})
}

// corsMiddleware handles CORS
func (s *Server) corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if origin := r.Header.Get("Origin"); origin != "" {
			for _, allowed := range s.config.AllowedOrigins {
				if origin == allowed {
					w.Header().Set("Access-Control-Allow-Origin", origin)
					break
				}
			}
		}
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == "OPTIONS" {
			return
		}

		next.ServeHTTP(w, r)
	})
}

// writeJSON writes a JSON response
func (s *Server) writeJSON(w http.ResponseWriter, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(data); err != nil {
		s.handleError(w, err, http.StatusInternalServerError)
	}
}

// handleError handles error responses
func (s *Server) handleError(w http.ResponseWriter, err interface{}, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	var msg string
	switch v := err.(type) {
	case error:
		msg = v.Error()
	case string:
		msg = v
	default:
		msg = fmt.Sprintf("%v", v)
	}
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

// Shutdown gracefully shuts down the server
func (s *Server) Shutdown(ctx context.Context) error {
	s.logger.Info("shutting down server")

	// Shutdown metrics server first
	if s.metricsServer != nil {
		if err := s.metricsServer.Shutdown(ctx); err != nil {
			s.logger.Error("error shutting down metrics server", zap.Error(err))
		}
	}

	// Add any other cleanup here

	return nil
}

// GetRouter returns the router for testing
func (s *Server) GetRouter() http.Handler {
	return s.router
}
