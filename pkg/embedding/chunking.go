package embedding

import (
	"fmt"
	"strings"
	"sync/atomic"
)

// ChunkingStrategy defines how to split large documents
type ChunkingStrategy int

const (
	// ChunkByParagraph splits text by paragraphs (double newlines)
	ChunkByParagraph ChunkingStrategy = iota
	// ChunkBySentence splits text by sentences
	ChunkBySentence
	// ChunkByWords splits text into word groups
	ChunkByWords
)

// ChunkingOptions configures how documents are split
type ChunkingOptions struct {
	// Strategy determines how to split the document
	Strategy ChunkingStrategy
	// MaxTokens is the maximum number of tokens per chunk
	MaxTokens int
	// PreserveWhitespace keeps original whitespace (default false)
	PreserveWhitespace bool
}

// DefaultChunkingOptions returns sensible defaults
func DefaultChunkingOptions() ChunkingOptions {
	return ChunkingOptions{
		Strategy:           ChunkByParagraph,
		MaxTokens:          512,
		PreserveWhitespace: false,
	}
}

// ChunkDocument splits a large document into smaller chunks based on the strategy
func (t *Tokenizer) ChunkDocument(text string, opts ChunkingOptions) ([]string, error) {
	if opts.MaxTokens <= 0 {
		opts.MaxTokens = int(atomic.LoadInt32(&t.seqLength))
	}

	// Normalize whitespace unless preservation is requested
	if !opts.PreserveWhitespace {
		text = normalizeWhitespace(text)
	}

	var initialChunks []string
	switch opts.Strategy {
	case ChunkByParagraph:
		initialChunks = splitByParagraph(text)
	case ChunkBySentence:
		initialChunks = splitBySentence(text)
	case ChunkByWords:
		initialChunks = splitByWords(text, 100) // Split into ~100 word chunks
	default:
		initialChunks = []string{text}
	}

	// Validate and merge/split chunks based on token count
	return t.validateAndMergeChunks(initialChunks, opts.MaxTokens)
}

// splitByParagraph splits text into paragraphs based on double newlines
func splitByParagraph(text string) []string {
	// Split on double newlines
	paragraphs := strings.Split(text, "\n\n")

	// Filter empty paragraphs and trim whitespace
	var chunks []string
	for _, p := range paragraphs {
		if p = strings.TrimSpace(p); p != "" {
			chunks = append(chunks, p)
		}
	}
	return chunks
}

// splitBySentence splits text into sentences using basic heuristics
func splitBySentence(text string) []string {
	var sentences []string
	var current strings.Builder
	var lastRune rune
	inQuote := false

	for _, r := range text {
		current.WriteRune(r)

		// Handle quotes to avoid splitting in the middle of quoted text
		if r == '"' {
			inQuote = !inQuote
		}

		// Check for sentence endings (.!?) but not in quotes or abbreviations
		if !inQuote && (r == '.' || r == '!' || r == '?') && lastRune != '.' {
			if current.Len() > 0 {
				sentences = append(sentences, strings.TrimSpace(current.String()))
				current.Reset()
			}
		}
		lastRune = r
	}

	// Add any remaining text
	if current.Len() > 0 {
		sentences = append(sentences, strings.TrimSpace(current.String()))
	}

	return sentences
}

// splitByWords splits text into chunks of approximately n words
func splitByWords(text string, n int) []string {
	var chunks []string
	words := strings.Fields(text)

	for i := 0; i < len(words); i += n {
		end := i + n
		if end > len(words) {
			end = len(words)
		}
		chunks = append(chunks, strings.Join(words[i:end], " "))
	}

	return chunks
}

// validateAndMergeChunks ensures chunks don't exceed max tokens and merges small chunks
func (t *Tokenizer) validateAndMergeChunks(chunks []string, maxTokens int) ([]string, error) {
	var result []string
	var current strings.Builder
	currentTokens := 0

	// Get tokenizer instance atomically
	tokenizer := t.tokenizer.Load()
	if tokenizer == nil {
		return nil, fmt.Errorf("tokenizer is not initialized or has been closed")
	}

	for _, chunk := range chunks {
		// Get token count for this chunk
		rawIds, _ := tokenizer.Encode(chunk, false)
		ids := make([]int64, len(rawIds))
		for i, id := range rawIds {
			ids[i] = int64(id)
		}

		// If chunk alone exceeds limit, split it into smaller pieces
		if len(ids) > maxTokens {
			// If we have accumulated text, add it first
			if current.Len() > 0 {
				result = append(result, strings.TrimSpace(current.String()))
				current.Reset()
				currentTokens = 0
			}

			// Split into smaller chunks by words
			words := strings.Fields(chunk)
			var subChunk strings.Builder

			for _, word := range words {
				// Try adding this word
				if subChunk.Len() > 0 {
					subChunk.WriteString(" ")
				}
				subChunk.WriteString(word)

				// Check token count
				rawIds, _ := tokenizer.Encode(subChunk.String(), false)
				ids := make([]int64, len(rawIds))
				for i, id := range rawIds {
					ids[i] = int64(id)
				}

				// If adding this word exceeded the limit, save the previous chunk and start a new one
				if len(ids) > maxTokens {
					// Remove the last word and space
					text := strings.TrimSpace(subChunk.String())
					text = strings.TrimSuffix(text, word)
					text = strings.TrimSpace(text)

					if text != "" {
						result = append(result, text)
					}

					// Start new chunk with the word that didn't fit
					subChunk.Reset()
					subChunk.WriteString(word)
				}
			}

			// Add any remaining text in the sub-chunk
			if subChunk.Len() > 0 {
				result = append(result, strings.TrimSpace(subChunk.String()))
			}
			continue
		}

		// If adding this chunk would exceed limit, start new chunk
		if currentTokens+len(ids) > maxTokens && current.Len() > 0 {
			result = append(result, strings.TrimSpace(current.String()))
			current.Reset()
			currentTokens = 0
		}

		// Add chunk to current
		if current.Len() > 0 {
			current.WriteString(" ")
		}
		current.WriteString(chunk)
		currentTokens += len(ids)
	}

	// Add any remaining text
	if current.Len() > 0 {
		result = append(result, strings.TrimSpace(current.String()))
	}

	return result, nil
}

// normalizeWhitespace replaces multiple whitespace characters with a single space
func normalizeWhitespace(text string) string {
	return strings.Join(strings.Fields(text), " ")
}
