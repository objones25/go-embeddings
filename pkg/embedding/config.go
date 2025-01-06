package embedding

import (
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/joho/godotenv"
)

// Config holds the configuration for the embedding service
type Config struct {
	// Model settings
	ModelPath string
	Tokenizer TokenizerConfig
	ModelType string
	Dimension int

	// Hardware acceleration
	EnableMetal   bool
	MetalDeviceID int

	// Performance settings
	BatchSize         int
	MaxSequenceLength int
	InterOpThreads    int
	IntraOpThreads    int

	// Memory management
	ModelMemoryLimit int64
	CacheSize        int
	PreloadModels    []string

	// Service settings
	RequestTimeout time.Duration

	// Advanced options
	Options Options
}

// Options contains additional configuration options
type Options struct {
	PadToMaxLength bool
	Normalize      bool
	CacheEnabled   bool
}

// MetalConfig contains Metal-specific settings
type MetalConfig struct {
	WorkgroupSize int
	TileSize      int
	VectorWidth   int
}

// GetModelDimension returns the embedding dimension for a given model ID
func GetModelDimension(modelID string) int {
	switch modelID {
	// MiniLM models
	case "sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1":
		return 384
	case "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2":
		return 384

	// MPNet models
	case "sentence-transformers/all-mpnet-base-v2":
		return 768

	// RoBERTa models
	case "sentence-transformers/all-roberta-large-v1":
		return 1024

	// E5 models
	case "intfloat/e5-large-v2":
		return 1024

	// BGE models
	case "BAAI/bge-large-en-v1.5":
		return 1024

	// Default case
	default:
		// Log a warning that we're using a default dimension
		fmt.Printf("WARNING: Unknown model ID %s, using default dimension of 768\n", modelID)
		return 768 // Default to 768 for most BERT-based models
	}
}

// LoadConfig loads configuration from environment variables
func LoadConfig() (*Config, error) {
	if err := godotenv.Load(); err != nil {
		return nil, fmt.Errorf("error loading .env file: %v", err)
	}

	modelID := os.Getenv("TOKENIZER_MODEL_ID")
	dimension := getEnvInt("EMBEDDING_DIMENSION", GetModelDimension(modelID))
	maxSeqLength := getEnvInt("MAX_SEQUENCE_LENGTH", 512)

	config := &Config{
		ModelPath: os.Getenv("MODEL_PATH"),
		Tokenizer: TokenizerConfig{
			LocalPath:      os.Getenv("TOKENIZER_PATH"),
			ModelID:        modelID,
			AuthToken:      os.Getenv("HF_AUTH_TOKEN"),
			CacheDir:       os.Getenv("TOKENIZER_CACHE_DIR"),
			SequenceLength: maxSeqLength,
		},
		Dimension:         dimension,
		EnableMetal:       getEnvBool("ENABLE_METAL", true),
		MetalDeviceID:     getEnvInt("METAL_DEVICE_ID", 0),
		BatchSize:         getEnvInt("BATCH_SIZE", 32),
		MaxSequenceLength: maxSeqLength,
		InterOpThreads:    getEnvInt("INTER_OP_THREADS", 1),
		IntraOpThreads:    getEnvInt("INTRA_OP_THREADS", 1),
		ModelMemoryLimit:  getEnvInt64("MODEL_MEMORY_LIMIT", 500*1024*1024), // 500MB default
		CacheSize:         getEnvInt("EMBEDDING_CACHE_SIZE", 10000),
		RequestTimeout:    getEnvDuration("REQUEST_TIMEOUT", 30*time.Second),
		Options: Options{
			PadToMaxLength: getEnvBool("PAD_TO_MAX_LENGTH", false),
			Normalize:      getEnvBool("NORMALIZE_EMBEDDINGS", true),
			CacheEnabled:   getEnvBool("ENABLE_CACHE", true),
		},
	}

	if err := config.validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %v", err)
	}

	return config, nil
}

// validate checks if the configuration is valid
func (c *Config) validate() error {
	if c.ModelPath == "" {
		return fmt.Errorf("model path is required")
	}
	if c.Tokenizer.LocalPath == "" && c.Tokenizer.ModelID == "" {
		return fmt.Errorf("either tokenizer path or model ID is required")
	}
	if c.BatchSize <= 0 {
		return fmt.Errorf("invalid batch size: %d", c.BatchSize)
	}
	if c.Dimension <= 0 {
		return fmt.Errorf("invalid embedding dimension: %d", c.Dimension)
	}
	return nil
}

// Helper functions to get environment variables with default values
func getEnvBool(key string, defaultVal bool) bool {
	if v, exists := os.LookupEnv(key); exists {
		b, err := strconv.ParseBool(v)
		if err != nil {
			return defaultVal
		}
		return b
	}
	return defaultVal
}

func getEnvInt(key string, defaultVal int) int {
	if v, exists := os.LookupEnv(key); exists {
		i, err := strconv.Atoi(v)
		if err != nil {
			return defaultVal
		}
		return i
	}
	return defaultVal
}

func getEnvInt64(key string, defaultVal int64) int64 {
	if v, exists := os.LookupEnv(key); exists {
		i, err := strconv.ParseInt(v, 10, 64)
		if err != nil {
			return defaultVal
		}
		return i
	}
	return defaultVal
}

func getEnvDuration(key string, defaultVal time.Duration) time.Duration {
	if v, exists := os.LookupEnv(key); exists {
		d, err := time.ParseDuration(v)
		if err != nil {
			return defaultVal
		}
		return d
	}
	return defaultVal
}
