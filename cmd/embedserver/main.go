package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
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
		log.Fatalf("Error loading config file: %v", err)
	}

	// Initialize logger
	logger, err := zap.NewProduction()
	if err != nil {
		log.Fatalf("Failed to create logger: %v", err)
	}
	defer logger.Sync()

	// Load embedding service configuration
	embeddingConfig, err := embedding.LoadConfig()
	if err != nil {
		logger.Fatal("Failed to load embedding config", zap.Error(err))
	}

	// Create embedding service
	ctx := context.Background()
	embeddingService, err := embedding.NewService(ctx, embeddingConfig)
	if err != nil {
		logger.Fatal("Failed to create embedding service", zap.Error(err))
	}
	defer embeddingService.Close()

	// Create server configuration
	serverConfig := &server.Config{
		Port:           getEnvOrDefault("PORT", "8080"),
		MetricsPort:    getEnvOrDefault("METRICS_PORT", "2112"),
		ReadTimeout:    getDurationOrDefault("READ_TIMEOUT", 30*time.Second),
		WriteTimeout:   getDurationOrDefault("WRITE_TIMEOUT", 30*time.Second),
		MaxRequestSize: getInt64OrDefault("MAX_REQUEST_SIZE", 1<<20), // 1MB
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
	logger.Info("Starting server...")
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
