package embedding

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"github.com/yalue/onnxruntime_go"
	"go.uber.org/zap"
)

// Result represents an embedding result
type Result struct {
	Text      string    `json:"text"`
	Embedding []float32 `json:"embedding"`
	Error     error     `json:"-"`
}

// Service defines the interface for the embedding service
type Service interface {
	// Embed generates embeddings for a single text
	Embed(ctx context.Context, text string) ([]float32, error)

	// BatchEmbed generates embeddings for multiple texts
	BatchEmbed(ctx context.Context, texts []string) ([][]float32, error)

	// BatchEmbedAsync generates embeddings asynchronously
	BatchEmbedAsync(ctx context.Context, texts []string, results chan<- Result, errors chan<- error) error

	// Close releases all resources
	Close() error
}

// service implements the Service interface
type service struct {
	config     *Config
	session    *onnxruntime_go.DynamicSession[int64, float32]
	tokenizer  *Tokenizer
	logger     *zap.Logger
	cache      *Cache
	workerPool chan struct{}
	mu         sync.RWMutex
}

var (
	initOnce sync.Once
	initErr  error
)

// getProjectRoot returns the absolute path to the project root directory
func getProjectRoot() (string, error) {
	// Get the current working directory
	cwd, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("failed to get working directory: %v", err)
	}

	// If we're in a test directory, go up to the project root
	if strings.Contains(cwd, "test/unit") || strings.Contains(cwd, "test/integration") {
		return filepath.Join(cwd, "../.."), nil
	}

	return cwd, nil
}

// initializeONNXRuntime initializes the ONNX Runtime environment once
func initializeONNXRuntime() error {
	initOnce.Do(func() {
		// Get the project root directory
		projectRoot, err := getProjectRoot()
		if err != nil {
			initErr = fmt.Errorf("failed to get project root: %v", err)
			return
		}

		// Set library path based on platform
		libPath := filepath.Join(projectRoot, "libs")
		if runtime.GOOS == "darwin" {
			onnxruntime_go.SetSharedLibraryPath(filepath.Join(libPath, "libonnxruntime.dylib"))
		} else {
			onnxruntime_go.SetSharedLibraryPath(filepath.Join(libPath, "onnxruntime.so"))
		}

		// Initialize ONNX runtime
		initErr = onnxruntime_go.InitializeEnvironment()
		if initErr != nil {
			initErr = fmt.Errorf("failed to initialize ONNX environment: %v", initErr)
			return
		}

		// Disable telemetry
		if err := onnxruntime_go.DisableTelemetry(); err != nil {
			initErr = fmt.Errorf("failed to disable telemetry: %v", err)
			return
		}
	})
	return initErr
}

// NewServiceWithLogger creates a new embedding service with a custom logger
func NewServiceWithLogger(ctx context.Context, config *Config, logger *zap.Logger) (Service, error) {
	// Validate batch size
	if config.BatchSize <= 0 {
		return nil, fmt.Errorf("invalid batch size: %d", config.BatchSize)
	}

	// Set dimension based on model if not already set
	if config.Dimension <= 0 && config.Tokenizer.ModelID != "" {
		config.Dimension = GetModelDimension(config.Tokenizer.ModelID)
	}

	// Initialize ONNX runtime
	if err := initializeONNXRuntime(); err != nil {
		return nil, err
	}

	// Create session options
	opts, err := onnxruntime_go.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %v", err)
	}
	defer opts.Destroy()

	// Configure session options
	opts.SetIntraOpNumThreads(config.InterOpThreads)
	opts.SetInterOpNumThreads(config.IntraOpThreads)

	// Enable hardware acceleration if requested and available
	if config.EnableMetal && runtime.GOOS == "darwin" {
		logger.Info("Enabling CoreML execution provider for hardware acceleration")

		// Calculate CoreML flags based on configuration
		var flags uint32
		if config.CoreMLConfig != nil {
			if config.CoreMLConfig.EnableCaching {
				flags |= 1 // COREML_FLAG_USE_CACHE
			}
			if config.CoreMLConfig.RequireANE {
				flags |= 2 // COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE
			}
		}

		if err := opts.AppendExecutionProviderCoreML(flags); err != nil {
			logger.Warn("Failed to enable CoreML execution provider, falling back to CPU",
				zap.Error(err),
				zap.Bool("caching_enabled", config.CoreMLConfig != nil && config.CoreMLConfig.EnableCaching),
				zap.Bool("ane_required", config.CoreMLConfig != nil && config.CoreMLConfig.RequireANE))
		} else {
			logger.Info("Successfully enabled CoreML execution provider",
				zap.Bool("caching_enabled", config.CoreMLConfig != nil && config.CoreMLConfig.EnableCaching),
				zap.Bool("ane_required", config.CoreMLConfig != nil && config.CoreMLConfig.RequireANE))
		}
	}

	// Create ONNX session
	session, err := onnxruntime_go.NewDynamicSession[int64, float32](
		config.ModelPath,
		[]string{"input_ids", "attention_mask", "token_type_ids"},
		[]string{"last_hidden_state", "pooler_output"},
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %v", err)
	}

	// Initialize tokenizer
	tokenizer, err := NewTokenizer(config.Tokenizer)
	if err != nil {
		session.Destroy()
		return nil, fmt.Errorf("failed to create tokenizer: %v", err)
	}

	// Initialize cache if enabled
	var cache *Cache
	if config.Options.CacheEnabled {
		cache = NewCache(config.CacheSize)
	}

	// Create worker pool
	workerPool := make(chan struct{}, config.BatchSize)

	s := &service{
		config:     config,
		session:    session,
		tokenizer:  tokenizer,
		logger:     logger,
		cache:      cache,
		workerPool: workerPool,
	}

	return s, nil
}

// NewService creates a new embedding service with default logger
func NewService(ctx context.Context, config *Config) (Service, error) {
	logger, err := zap.NewProduction()
	if err != nil {
		return nil, fmt.Errorf("failed to create logger: %v", err)
	}
	return NewServiceWithLogger(ctx, config, logger)
}

// Embed generates embeddings for a single text
func (s *service) Embed(ctx context.Context, text string) ([]float32, error) {
	if text == "" {
		return nil, fmt.Errorf("empty text")
	}

	// Check for context cancellation
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// Check cache first if enabled
	if s.cache != nil {
		if embedding, found := s.cache.Get(text); found {
			return embedding, nil
		}
	}

	// Check context again before expensive tokenization
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// Tokenize text
	tokens, err := s.tokenizer.Tokenize(text, s.config.MaxSequenceLength)
	if err != nil {
		return nil, fmt.Errorf("failed to tokenize text: %w", err)
	}

	// Create attention mask (1 for all tokens)
	attentionMask := make([]int64, len(tokens))
	for i := range attentionMask {
		attentionMask[i] = 1
	}

	// Create token type IDs (0 for all tokens)
	tokenTypeIDs := make([]int64, len(tokens))

	// Check context before tensor creation
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// Create input tensors
	inputTensor, err := onnxruntime_go.NewTensor([]int64{1, int64(len(tokens))}, tokens)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	attentionTensor, err := onnxruntime_go.NewTensor([]int64{1, int64(len(attentionMask))}, attentionMask)
	if err != nil {
		return nil, fmt.Errorf("failed to create attention mask tensor: %w", err)
	}
	defer attentionTensor.Destroy()

	tokenTypeTensor, err := onnxruntime_go.NewTensor([]int64{1, int64(len(tokenTypeIDs))}, tokenTypeIDs)
	if err != nil {
		return nil, fmt.Errorf("failed to create token type tensor: %w", err)
	}
	defer tokenTypeTensor.Destroy()

	// fmt.Printf("DEBUG Service: Input dimensions:\n  - Sequence length: %d\n  - Input data: %v\n  - Attention mask: %v\n  - Token type IDs: %v\n",
	// 	len(tokens), tokens, attentionMask, tokenTypeIDs)

	// Create output tensors
	hiddenTensor, err := onnxruntime_go.NewEmptyTensor[float32]([]int64{1, int64(len(tokens)), int64(s.config.Dimension)})
	if err != nil {
		return nil, fmt.Errorf("failed to create hidden tensor: %w", err)
	}
	defer hiddenTensor.Destroy()

	poolerTensor, err := onnxruntime_go.NewEmptyTensor[float32]([]int64{1, int64(s.config.Dimension)})
	if err != nil {
		return nil, fmt.Errorf("failed to create pooler tensor: %w", err)
	}
	defer poolerTensor.Destroy()

	// Check context before running inference
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// Run inference with a goroutine to support cancellation
	errCh := make(chan error, 1)
	go func() {
		errCh <- s.session.Run(
			[]*onnxruntime_go.Tensor[int64]{inputTensor, attentionTensor, tokenTypeTensor},
			[]*onnxruntime_go.Tensor[float32]{hiddenTensor, poolerTensor},
		)
	}()

	select {
	case err := <-errCh:
		if err != nil {
			return nil, fmt.Errorf("failed to run inference: %w", err)
		}
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	// Get embedding from pooler output
	embedding := make([]float32, s.config.Dimension)
	copy(embedding, poolerTensor.GetData())

	if s.config.Options.Normalize {
		normalizeVector(embedding)
	}

	// Cache result if enabled
	if s.cache != nil {
		s.cache.Set(text, embedding)
	}

	return embedding, nil
}

// BatchEmbed generates embeddings for multiple texts
func (s *service) BatchEmbed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("empty text array")
	}

	// Validate individual texts
	for i, text := range texts {
		if text == "" {
			return nil, fmt.Errorf("empty text at index %d", i)
		}
	}

	// Process texts in batches
	results := make([][]float32, 0, len(texts))
	for i := 0; i < len(texts); i += s.config.BatchSize {
		// Check context before processing each batch
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		end := i + s.config.BatchSize
		if end > len(texts) {
			end = len(texts)
		}
		batch := texts[i:end]

		// Process each batch
		batchResults, err := s.processBatch(ctx, batch)
		if err != nil {
			return nil, fmt.Errorf("batch processing failed: %v", err)
		}

		results = append(results, batchResults...)
	}

	return results, nil
}

// BatchEmbedAsync generates embeddings asynchronously
func (s *service) BatchEmbedAsync(ctx context.Context, texts []string, results chan<- Result, errors chan<- error) error {
	if len(texts) == 0 {
		return fmt.Errorf("empty text array")
	}

	// Validate individual texts
	for i, text := range texts {
		if text == "" {
			return fmt.Errorf("empty text at index %d", i)
		}
	}

	// Process texts in batches
	for i := 0; i < len(texts); i += s.config.BatchSize {
		// Check context before starting each batch
		if err := ctx.Err(); err != nil {
			return err
		}

		end := i + s.config.BatchSize
		if end > len(texts) {
			end = len(texts)
		}
		batch := texts[i:end]

		// Process each batch in a goroutine
		go func(batch []string, startIdx int) {
			// Process the batch
			batchResults, err := s.processBatch(ctx, batch)
			if err != nil {
				select {
				case errors <- fmt.Errorf("batch processing failed: %v", err):
				case <-ctx.Done():
				}
				return
			}

			// Send results to channel
			for j, embedding := range batchResults {
				select {
				case <-ctx.Done():
					return
				case results <- Result{
					Text:      batch[j],
					Embedding: embedding,
				}:
				}
			}
		}(batch, i)
	}

	return nil
}

// processBatch processes a batch of texts and returns their embeddings
func (s *service) processBatch(ctx context.Context, batch []string) ([][]float32, error) {
	// Acquire worker from pool
	select {
	case s.workerPool <- struct{}{}:
		defer func() { <-s.workerPool }()
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	// Check context before tokenization
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// Tokenize all texts in batch
	tokens, err := s.tokenizer.TokenizeBatch(batch, s.config.MaxSequenceLength)
	if err != nil {
		return nil, fmt.Errorf("failed to tokenize batch: %v", err)
	}

	// Check context before tensor creation
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// Determine sequence length
	seqLen := 0
	if s.config.Options.PadToMaxLength {
		seqLen = s.config.MaxSequenceLength
	} else {
		// Find max length in batch
		for _, seq := range tokens {
			if len(seq) > seqLen {
				seqLen = len(seq)
			}
		}
	}

	// Convert tokens to input tensors
	batchSize := len(batch)
	inputData := make([]int64, batchSize*seqLen)
	attentionMask := make([]int64, batchSize*seqLen)
	tokenTypeIds := make([]int64, batchSize*seqLen)

	// Fill input data and attention mask
	for i, tokenList := range tokens {
		for j := 0; j < seqLen; j++ {
			idx := i*seqLen + j
			if j < len(tokenList) {
				inputData[idx] = tokenList[j]
				attentionMask[idx] = 1
			} else {
				inputData[idx] = 0 // Padding token ID
				attentionMask[idx] = 0
			}
			tokenTypeIds[idx] = 0 // Always 0 for single sequence
		}
	}

	// Create input tensors
	shape := onnxruntime_go.NewShape(int64(batchSize), int64(seqLen))
	inputTensor, err := onnxruntime_go.NewTensor(shape, inputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %v", err)
	}
	defer inputTensor.Destroy()

	attentionTensor, err := onnxruntime_go.NewTensor(shape, attentionMask)
	if err != nil {
		return nil, fmt.Errorf("failed to create attention tensor: %v", err)
	}
	defer attentionTensor.Destroy()

	tokenTypeTensor, err := onnxruntime_go.NewTensor(shape, tokenTypeIds)
	if err != nil {
		return nil, fmt.Errorf("failed to create token type tensor: %v", err)
	}
	defer tokenTypeTensor.Destroy()

	// Create output tensors
	hiddenShape := onnxruntime_go.NewShape(int64(batchSize), int64(seqLen), int64(s.config.Dimension))
	poolerShape := onnxruntime_go.NewShape(int64(batchSize), int64(s.config.Dimension))

	hiddenTensor, err := onnxruntime_go.NewEmptyTensor[float32](hiddenShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create hidden tensor: %v", err)
	}
	defer hiddenTensor.Destroy()

	poolerTensor, err := onnxruntime_go.NewEmptyTensor[float32](poolerShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create pooler tensor: %v", err)
	}
	defer poolerTensor.Destroy()

	// Run inference with a goroutine to support cancellation
	errCh := make(chan error, 1)
	go func() {
		errCh <- s.session.Run(
			[]*onnxruntime_go.Tensor[int64]{inputTensor, attentionTensor, tokenTypeTensor},
			[]*onnxruntime_go.Tensor[float32]{hiddenTensor, poolerTensor},
		)
	}()

	select {
	case err := <-errCh:
		if err != nil {
			return nil, fmt.Errorf("failed to run inference: %w", err)
		}
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	// Get embeddings from pooler output
	embeddings := make([][]float32, batchSize)
	poolerData := poolerTensor.GetData()

	for i := 0; i < batchSize; i++ {
		start := i * s.config.Dimension
		end := start + s.config.Dimension
		embedding := make([]float32, s.config.Dimension)
		copy(embedding, poolerData[start:end])
		if s.config.Options.Normalize {
			normalizeVector(embedding)
		}
		embeddings[i] = embedding

		// Cache result if enabled
		if s.cache != nil {
			s.cache.Set(batch[i], embedding)
		}
	}

	return embeddings, nil
}

// Close releases all resources
func (s *service) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.session != nil {
		s.session.Destroy()
		s.session = nil
	}

	if s.logger != nil {
		_ = s.logger.Sync()
	}

	return nil
}

// normalizeVector normalizes a vector in-place using L2 normalization
func normalizeVector(v []float32) {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	norm := float32(1.0 / float32(math.Sqrt(float64(sum))))
	for i := range v {
		v[i] *= norm
	}
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
