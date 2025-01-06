package unit

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"testing"

	"github.com/objones25/go-embeddings/pkg/embedding"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func init() {
	// Enable debug logging
	log.SetFlags(log.Lshortfile | log.LstdFlags)

	// Set up library paths
	workspaceDir, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		log.Printf("Error getting workspace dir: %v", err)
	}
	libsDir := filepath.Join(workspaceDir, "libs")

	// Set environment variables
	os.Setenv("DYLD_LIBRARY_PATH", libsDir)
	os.Setenv("LD_LIBRARY_PATH", libsDir)
	os.Setenv("ONNXRUNTIME_LIB_PATH", filepath.Join(libsDir, "libonnxruntime.1.20.0.dylib"))
	os.Setenv("CGO_LDFLAGS", fmt.Sprintf("-L%s", libsDir))

	// Log test data paths
	testDataPath := filepath.Join("..", "..", "testdata")
	log.Printf("Test data path: %s", testDataPath)
	if _, err := os.Stat(testDataPath); os.IsNotExist(err) {
		log.Printf("Warning: Test data directory does not exist")
	}

	tokenizerPath := filepath.Join(testDataPath, "tokenizer.json")
	log.Printf("Tokenizer path: %s (exists: %v)", tokenizerPath, fileExists(tokenizerPath))
}

func TestNewTokenizer(t *testing.T) {
	testDataPath := filepath.Join("..", "..", "testdata")

	tests := []struct {
		name        string
		config      embedding.TokenizerConfig
		expectError bool
	}{
		{
			name: "valid local path",
			config: embedding.TokenizerConfig{
				LocalPath:      filepath.Join(testDataPath, "tokenizer.json"),
				SequenceLength: 512,
			},
			expectError: false,
		},
		{
			name: "invalid local path",
			config: embedding.TokenizerConfig{
				LocalPath:      "nonexistent.json",
				SequenceLength: 512,
			},
			expectError: true,
		},
		{
			name: "valid model ID",
			config: embedding.TokenizerConfig{
				ModelID:        "sentence-transformers/all-MiniLM-L6-v2",
				SequenceLength: 512,
			},
			expectError: false,
		},
		{
			name:        "no path or model ID",
			config:      embedding.TokenizerConfig{},
			expectError: true,
		},
		{
			name: "invalid sequence length",
			config: embedding.TokenizerConfig{
				LocalPath:      filepath.Join(testDataPath, "tokenizer.json"),
				SequenceLength: 0,
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			log.Printf("Running test case: %s with config: %+v", tt.name, tt.config)
			tokenizer, err := embedding.NewTokenizer(tt.config)
			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, tokenizer)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, tokenizer)
			}
		})
	}
}

func TestTokenize(t *testing.T) {
	// Create test tokenizer file
	tokenizer, err := embedding.NewTokenizer(embedding.TokenizerConfig{
		ModelID:        "sentence-transformers/all-MiniLM-L6-v2",
		SequenceLength: 512,
	})
	require.NoError(t, err)
	defer tokenizer.Close()

	tests := []struct {
		name           string
		text           string
		maxLength      int
		expectError    bool
		expectedLength int
		expectedPadded bool
	}{
		{
			name:           "valid text",
			text:           "Hello, world!",
			maxLength:      512,
			expectError:    false,
			expectedLength: 512, // Full sequence with padding
			expectedPadded: true,
		},
		{
			name:           "empty text",
			text:           "",
			maxLength:      512,
			expectError:    false,
			expectedLength: 512, // Full sequence with padding
			expectedPadded: true,
		},
		{
			name:           "long text",
			text:           "This is a very long text that should be truncated",
			maxLength:      512,
			expectError:    false,
			expectedLength: 512, // Full sequence with padding
			expectedPadded: true,
		},
		{
			name:           "special characters",
			text:           "!@#$%^&*()_+{}|:\"<>?~`-=[]\\;',./",
			maxLength:      512,
			expectError:    false,
			expectedLength: 512, // Full sequence with padding
			expectedPadded: true,
		},
		{
			name:           "multilingual text",
			text:           "Hello 你好 Bonjour こんにちは Hola",
			maxLength:      512,
			expectError:    false,
			expectedLength: 512, // Full sequence with padding
			expectedPadded: true,
		},
		{
			name:           "whitespace only",
			text:           "     \t\n\r",
			maxLength:      512,
			expectError:    false,
			expectedLength: 512, // Full sequence with padding
			expectedPadded: true,
		},
		{
			name:           "repeated characters",
			text:           "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
			maxLength:      512,
			expectError:    false,
			expectedLength: 512, // Full sequence with padding
			expectedPadded: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			log.Printf("Running test case: %s with text: %q", tt.name, tt.text)
			tokens, err := tokenizer.Tokenize(tt.text, tt.maxLength)
			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, tokens)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, tokens)
				assert.Equal(t, tt.expectedLength, len(tokens),
					"Expected token length %d but got %d for text: %q",
					tt.expectedLength, len(tokens), tt.text)

				// Check padding
				if tt.expectedPadded {
					// Count non-padding tokens
					nonPadding := 0
					for _, token := range tokens {
						if token != 0 {
							nonPadding++
						}
					}
					assert.Less(t, nonPadding, tt.expectedLength,
						"Expected padding in sequence but found none")
				}

				log.Printf("Got %d tokens for text: %q (%d non-padding)",
					len(tokens), tt.text, func() int {
						count := 0
						for _, token := range tokens {
							if token != 0 {
								count++
							}
						}
						return count
					}())
			}
		})
	}
}

func TestTokenizeBatch(t *testing.T) {
	tokenizer, err := embedding.NewTokenizer(embedding.TokenizerConfig{
		ModelID:        "sentence-transformers/all-MiniLM-L6-v2",
		SequenceLength: 512,
	})
	require.NoError(t, err)
	defer tokenizer.Close()

	tests := []struct {
		name          string
		texts         []string
		maxLength     int
		expectError   bool
		expectedSizes struct {
			sequences  int   // Number of sequences
			seqLength  int   // Length of each sequence
			nonPadding []int // Expected non-padding tokens per sequence
		}
	}{
		{
			name: "valid batch",
			texts: []string{
				"Hello, world!",
				"This is a test.",
			},
			maxLength:   512,
			expectError: false,
			expectedSizes: struct {
				sequences  int
				seqLength  int
				nonPadding []int
			}{
				sequences:  2,
				seqLength:  512,
				nonPadding: []int{6, 7}, // [CLS] tokens [SEP]
			},
		},
		{
			name:        "empty batch",
			texts:       []string{},
			maxLength:   512,
			expectError: false,
			expectedSizes: struct {
				sequences  int
				seqLength  int
				nonPadding []int
			}{
				sequences:  0,
				seqLength:  512,
				nonPadding: []int{},
			},
		},
		{
			name: "batch with empty text",
			texts: []string{
				"Hello",
				"",
				"World",
			},
			maxLength:   512,
			expectError: false,
			expectedSizes: struct {
				sequences  int
				seqLength  int
				nonPadding []int
			}{
				sequences:  3,
				seqLength:  512,
				nonPadding: []int{3, 2, 3}, // [CLS] token [SEP]
			},
		},
		{
			name: "batch with varying lengths",
			texts: []string{
				"Short text",
				"This is a longer piece of text that should still be padded",
				"Medium length text here",
			},
			maxLength:   512,
			expectError: false,
			expectedSizes: struct {
				sequences  int
				seqLength  int
				nonPadding []int
			}{
				sequences:  3,
				seqLength:  512,
				nonPadding: []int{4, 14, 6}, // Updated token counts based on actual tokenization
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			log.Printf("Running test case: %s with %d texts", tt.name, len(tt.texts))
			tokens, err := tokenizer.TokenizeBatch(tt.texts, tt.maxLength)
			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, tokens)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, tokens)

				// Check number of sequences
				assert.Equal(t, tt.expectedSizes.sequences, len(tokens),
					"Expected %d sequences, got %d", tt.expectedSizes.sequences, len(tokens))

				// Check each sequence
				for i, tokenList := range tokens {
					// Check sequence length
					assert.Equal(t, tt.expectedSizes.seqLength, len(tokenList),
						"Sequence %d: expected length %d, got %d",
						i, tt.expectedSizes.seqLength, len(tokenList))

					// Count non-padding tokens
					nonPadding := 0
					for _, token := range tokenList {
						if token != 0 {
							nonPadding++
						}
					}

					// Check non-padding token count
					if i < len(tt.expectedSizes.nonPadding) {
						assert.Equal(t, tt.expectedSizes.nonPadding[i], nonPadding,
							"Sequence %d: expected %d non-padding tokens, got %d",
							i, tt.expectedSizes.nonPadding[i], nonPadding)
					}

					log.Printf("Sequence %d: %d total tokens, %d non-padding tokens",
						i, len(tokenList), nonPadding)
				}
			}
		})
	}
}
