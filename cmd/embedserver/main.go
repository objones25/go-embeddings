package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"runtime"
	"strconv"
	"syscall"
	"time"

	"github.com/joho/godotenv"
	"go.uber.org/zap"

	"github.com/objones25/go-embeddings/pkg/embedding"
	"github.com/objones25/go-embeddings/pkg/server"
)

func main() {
	// Parse command line flags
	configPath := flag.String("config", ".env", "Path to config file")
	flag.Parse()

	// Load environment variables
	if err := godotenv.Load(*configPath); err != nil {
		log.Printf("Warning: Error loading config file: %v", err)
	}

	// Initialize logger
	logger, err := zap.NewProduction()
	if err != nil {
		log.Fatalf("Failed to create logger: %v", err)
	}
	defer logger.Sync()

	// Optimize GOMAXPROCS based on available CPU cores
	runtime.GOMAXPROCS(runtime.NumCPU())

	// Load embedding service configuration with optimized defaults
	embeddingConfig := &embedding.Config{
		// Model settings
		ModelPath:         getEnvOrDefault("MODEL_PATH", "models/all-MiniLM-L6-v2"),
		ModelMemoryLimit:  getInt64OrDefault("MODEL_MEMORY_LIMIT", 500*1024*1024), // 500MB
		MaxSequenceLength: getIntOrDefault("MAX_SEQUENCE_LENGTH", 512),
		Dimension:         384, // Default for MiniLM-L6

		// Batch processing settings (based on benchmarks)
		BatchSize:      getIntOrDefault("BATCH_SIZE", 32), // Optimal for most cases
		InterOpThreads: getIntOrDefault("INTER_OP_THREADS", 1),
		IntraOpThreads: getIntOrDefault("INTRA_OP_THREADS", 1),

		// Cache settings
		CacheSize:        getIntOrDefault("CACHE_SIZE", 10000),         // Number of embeddings
		CacheSizeBytes:   getInt64OrDefault("CACHE_SIZE_BYTES", 1<<30), // 1GB
		DiskCacheEnabled: getBoolOrDefault("DISK_CACHE_ENABLED", true),
		DiskCachePath:    getEnvOrDefault("DISK_CACHE_PATH", "./cache"),
		CacheWarming:     getBoolOrDefault("ENABLE_CACHE_WARMING", true),
		WarmingInterval:  getDurationOrDefault("CACHE_WARMING_INTERVAL", 5*time.Minute),

		// Hardware acceleration
		EnableMetal: getBoolOrDefault("ENABLE_METAL", true),
		CoreMLConfig: &embedding.CoreMLConfig{
			EnableCaching: getBoolOrDefault("COREML_CACHING", true),
			RequireANE:    getBoolOrDefault("REQUIRE_ANE", false),
		},

		// Request handling
		RequestTimeout: getDurationOrDefault("REQUEST_TIMEOUT", 30*time.Second),

		// Additional options
		Options: embedding.Options{
			PadToMaxLength: getBoolOrDefault("PAD_TO_MAX_LENGTH", false),
			Normalize:      getBoolOrDefault("NORMALIZE_EMBEDDINGS", true),
			CacheEnabled:   getBoolOrDefault("ENABLE_CACHE", true),
		},
	}

	// Create embedding service
	ctx := context.Background()
	embeddingService, err := embedding.NewService(ctx, embeddingConfig)
	if err != nil {
		logger.Fatal("Failed to create embedding service", zap.Error(err))
	}
	defer embeddingService.Close()

	// Create server configuration with optimized timeouts
	serverConfig := &server.Config{
		Port:           getEnvOrDefault("PORT", "8080"),
		MetricsPort:    getEnvOrDefault("METRICS_PORT", "2112"),
		ReadTimeout:    getDurationOrDefault("READ_TIMEOUT", 30*time.Second),
		WriteTimeout:   getDurationOrDefault("WRITE_TIMEOUT", 30*time.Second),
		MaxRequestSize: getInt64OrDefault("MAX_REQUEST_SIZE", 5<<20), // 5MB
		AllowedOrigins: []string{"*"},                                // Configure as needed
	}

	// Create HTTP server
	srv, err := server.NewServer(serverConfig, embeddingService)
	if err != nil {
		logger.Fatal("Failed to create server", zap.Error(err))
	}

	// Handle graceful shutdown
	done := make(chan bool)
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-quit
		logger.Info("Server is shutting down...")

		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		if err := srv.Shutdown(ctx); err != nil {
			logger.Error("Failed to gracefully shutdown server", zap.Error(err))
		}
		close(done)
	}()

	// Start server
	logger.Info("Starting server...",
		zap.String("port", serverConfig.Port),
		zap.String("metrics_port", serverConfig.MetricsPort),
		zap.Int("batch_size", embeddingConfig.BatchSize),
		zap.Bool("metal_enabled", embeddingConfig.EnableMetal),
	)

	if err := srv.Start(); err != nil {
		logger.Error("Server failed", zap.Error(err))
	}

	<-done
	logger.Info("Server stopped")
}

// Helper functions for environment variables
func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getDurationOrDefault(key string, defaultValue time.Duration) time.Duration {
	if value := os.Getenv(key); value != "" {
		if duration, err := time.ParseDuration(value); err == nil {
			return duration
		}
	}
	return defaultValue
}

func getInt64OrDefault(key string, defaultValue int64) int64 {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.ParseInt(value, 10, 64); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func getIntOrDefault(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func getBoolOrDefault(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		if boolValue, err := strconv.ParseBool(value); err == nil {
			return boolValue
		}
	}
	return defaultValue
}
