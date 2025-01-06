package embedding

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/daulet/tokenizers"
)

// TokenizerConfig holds configuration for the tokenizer
type TokenizerConfig struct {
	// Path to local tokenizer file, if using a local model
	LocalPath string

	// HuggingFace model ID, if using a pre-trained model
	ModelID string

	// Optional auth token for private models
	AuthToken string

	// Optional cache directory for downloaded models
	CacheDir string

	// Required sequence length for all outputs
	SequenceLength int
}

// Tokenizer wraps the HuggingFace tokenizer
type Tokenizer struct {
	tokenizer atomic.Pointer[tokenizers.Tokenizer] // Protected by atomic operations
	seqLength int32                                // Protected by atomic operations
	mu        sync.RWMutex                         // Still needed for complex operations
}

// NewTokenizer creates a new tokenizer from either a local file or pre-trained model
func NewTokenizer(config TokenizerConfig) (*Tokenizer, error) {
	if config.SequenceLength <= 0 {
		return nil, fmt.Errorf("SequenceLength must be positive, got %d", config.SequenceLength)
	}

	var t *tokenizers.Tokenizer
	var err error

	if config.LocalPath != "" {
		// Load from local file
		t, err = tokenizers.FromFile(config.LocalPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load tokenizer from %s: %v", config.LocalPath, err)
		}
	} else if config.ModelID != "" {
		// Load pre-trained model
		opts := []tokenizers.TokenizerConfigOption{}
		if config.AuthToken != "" {
			opts = append(opts, tokenizers.WithAuthToken(config.AuthToken))
		}
		if config.CacheDir != "" {
			opts = append(opts, tokenizers.WithCacheDir(config.CacheDir))
		}

		t, err = tokenizers.FromPretrained(config.ModelID, opts...)
		if err != nil {
			return nil, fmt.Errorf("failed to load pre-trained tokenizer %s: %v", config.ModelID, err)
		}
	} else {
		return nil, fmt.Errorf("either LocalPath or ModelID must be provided")
	}

	tok := &Tokenizer{}
	tok.tokenizer.Store(t)
	atomic.StoreInt32(&tok.seqLength, int32(config.SequenceLength))
	return tok, nil
}

// tokenizeInternal performs tokenization without mutex locking
func (t *Tokenizer) tokenizeInternal(text string) ([]int64, error) {
	// Get tokenizer instance atomically
	tokenizer := t.tokenizer.Load()
	if tokenizer == nil {
		return nil, fmt.Errorf("tokenizer is not initialized or has been closed")
	}
	seqLen := atomic.LoadInt32(&t.seqLength)

	// First get token count without truncation to check length
	rawIds, _ := tokenizer.Encode(text, false)

	// Convert uint32 slice to int64 slice
	ids := make([]int64, len(rawIds))
	for i, id := range rawIds {
		ids[i] = int64(id)
	}

	// If text exceeds maximum length, try chunking
	if len(ids) > int(seqLen) {
		chunks, err := t.ChunkDocument(text, DefaultChunkingOptions())
		if err != nil {
			return nil, fmt.Errorf("failed to chunk long text: %v", err)
		}
		if len(chunks) == 0 {
			return nil, fmt.Errorf("chunking produced no valid chunks")
		}
		// Use the first chunk for this request
		text = chunks[0]
		// Re-encode with the first chunk
		rawIds, _ = tokenizer.Encode(text, false)
		ids = make([]int64, len(rawIds))
		for i, id := range rawIds {
			ids[i] = int64(id)
		}
	}

	// Now encode with truncation for the actual result
	rawTruncatedIds, _ := tokenizer.Encode(text, true)
	truncatedIds := make([]int64, len(rawTruncatedIds))
	for i, id := range rawTruncatedIds {
		truncatedIds[i] = int64(id)
	}

	// Create result slice with fixed length
	result := make([]int64, seqLen)

	// Copy tokens, padding with zeros
	n := len(truncatedIds)
	if n > int(seqLen) {
		n = int(seqLen)
	}
	copy(result[:n], truncatedIds[:n])
	// Rest of slice is already zero-padded

	return result, nil
}

// Tokenize converts a text into a sequence of token IDs with consistent length
func (t *Tokenizer) Tokenize(text string, _ int) ([]int64, error) {
	// No need for mutex lock since we use atomic operations
	return t.tokenizeInternal(text)
}

// TokenizeBatch converts multiple texts into sequences of token IDs with consistent length
func (t *Tokenizer) TokenizeBatch(texts []string, _ int) ([][]int64, error) {
	// Create channels for results and errors
	type result struct {
		index int
		ids   []int64
		err   error
	}
	results := make(chan result, len(texts))

	// Process texts concurrently with bounded concurrency
	maxWorkers := runtime.GOMAXPROCS(0)
	sem := make(chan struct{}, maxWorkers)

	// Launch workers
	var wg sync.WaitGroup
	for i, text := range texts {
		wg.Add(1)
		go func(i int, text string) {
			defer wg.Done()

			// Acquire semaphore
			sem <- struct{}{}
			defer func() { <-sem }()

			// Process the text (no mutex needed)
			ids, err := t.tokenizeInternal(text)
			results <- result{index: i, ids: ids, err: err}
		}(i, text)
	}

	// Wait for all workers to complete
	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results in order
	tokenized := make([][]int64, len(texts))
	for r := range results {
		if r.err != nil {
			return nil, fmt.Errorf("failed to tokenize text at index %d: %w", r.index, r.err)
		}
		tokenized[r.index] = r.ids
	}

	return tokenized, nil
}

// Close releases tokenizer resources
func (t *Tokenizer) Close() error {
	// Use mutex for complex cleanup operation
	t.mu.Lock()
	defer t.mu.Unlock()

	// Atomically set tokenizer to nil
	t.tokenizer.Store(nil)
	return nil
}
