I'll create a complete, standalone README for the go-embeddings library. Let me first structure this properly.



```markdown
# Go Embeddings Library

A high-performance, production-ready embedding service for Go, supporting multiple transformer models with MacOS Metal acceleration.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Supported Models](#supported-models)
- [Usage](#usage)
- [Performance Optimization](#performance-optimization)
- [Development](#development)
- [Testing](#testing)
- [Benchmarking](#benchmarking)
- [Contributing](#contributing)
- [License](#license)

## Features

### Core Features
- Production-ready transformer-based text embeddings
- MacOS Metal/CoreML hardware acceleration with ANE support
- Multi-model support with dynamic loading
- Optimized batch processing with auto-tuning
- Thread-safe implementation
- Comprehensive error handling and logging
- Built-in performance monitoring
- Memory-efficient model management

### Advanced Features
- Dynamic batch size optimization based on input length and latency
- Two-level caching system (memory + disk) with LRU eviction
- Automatic cache warming and invalidation
- Prometheus metrics integration
- Graceful shutdown and resource cleanup
- Support for both sync and async operations
- Chunking support for long documents
- Parallel processing with worker pools

### Performance Features
- Auto-tuning batch sizes based on input characteristics
- CoreML acceleration with caching support
- Memory-efficient token processing
- Zero-copy optimizations where possible
- Configurable thread management
- Built-in performance profiling

## Requirements

### System Requirements
- Go 1.21 or later
- MacOS ARM64 (M-series chips) or x86_64
- Minimum 8GB RAM (16GB recommended for production)
- 2GB free disk space for models

### Dependencies
- ONNX Runtime v1.20.0
- HuggingFace Tokenizers library
- Python 3.8+ (for model conversion only)

## Installation

### Quick Start
```bash
# Clone the repository
git clone https://github.com/objones25/go-embeddings
cd go-embeddings

# Run the automated setup script
make setup

# Run tests to verify installation
make test
```

### Manual Installation

1. Set up the project structure:
```bash
mkdir -p go-embeddings/{cmd,internal,pkg,models,libs,configs,scripts}
cd go-embeddings
go mod init github.com/objones25/go-embeddings
```

2. Install dependencies using our automated script:
```bash
./scripts/install_deps.sh
```

3. Set up the environment:
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
vim .env
```

4. Install Python dependencies for model conversion:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r scripts/requirements.txt
```

5. Download and convert models:
```bash
# Convert all supported models
make models

# Or convert specific models
python scripts/convert_model.py --model all-MiniLM-L6-v2 --output-dir models
```

### Docker Installation
```bash
# Build and run with Docker
docker-compose up -d

# Or for development environment
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

## Performance Optimization

### Memory Management
```go
// Configure memory settings
config := &embedding.Config{
    ModelMemoryLimit: 500 * 1024 * 1024,  // 500MB
    CacheSize:        10000,              // Number of embeddings to cache
    CacheSizeBytes:   1 << 30,            // 1GB cache size limit
    DiskCacheEnabled: true,
    DiskCachePath:    "./cache",
}
```

### Batch Processing Optimization
The library includes an intelligent dynamic batcher that automatically adjusts batch sizes based on:
- Input text length
- Processing latency
- System load
- Memory usage

```go
// Configure batch processing
config := &embedding.Config{
    BatchSize:         32,    // Initial batch size
    MaxSequenceLength: 512,   // Maximum sequence length
    Options: embedding.Options{
        Normalize:    true,
        CacheEnabled: true,
    },
}
```

### Hardware Acceleration
CoreML acceleration on Apple Silicon:
```go
config := &embedding.Config{
    EnableMetal: true,
    CoreMLConfig: &embedding.CoreMLConfig{
        EnableCaching: true,  // Cache compiled CoreML models
        RequireANE:    false, // Require Apple Neural Engine
    },
}
```

### Performance Characteristics

#### Single Embedding Performance
| Mode   | Latency | Memory/op | Allocs/op |
|--------|---------|-----------|-----------|
| CPU    | 1.85μs  | 1.5 KB    | 1         |
| CoreML | 1.86μs  | 1.5 KB    | 1         |

#### Batch Processing Performance
| Batch Size | Mode   | Latency | Memory/op | Allocs/op |
|------------|--------|---------|-----------|-----------|
| 32 inputs  | CPU    | 88.5ms  | 3.2 MB    | 1,113     |
| 32 inputs  | CoreML | 87.8ms  | 3.2 MB    | 1,113     |
| 128 inputs | CPU    | 374ms   | 13.1 MB   | 4,345     |
| 128 inputs | CoreML | 369ms   | 13.1 MB   | 4,345     |
| 256 inputs | CPU    | 775ms   | 26.2 MB   | 8,652     |
| 256 inputs | CoreML | 778ms   | 26.2 MB   | 8,650     |

#### Cached Performance
| Mode   | Latency | Memory/op | Allocs/op |
|--------|---------|-----------|-----------|
| CPU    | 1.90μs  | 1.5 KB    | 1         |
| CoreML | 1.88μs  | 1.5 KB    | 1         |

### Monitoring and Metrics
The service exposes Prometheus metrics at `:2112/metrics`:
- Request latencies
- Batch sizes
- Cache hit rates
- Memory usage
- Error rates

### Optimization Tips
1. **Batch Size Tuning**
   - Start with batch size 32
   - Let auto-tuning adjust based on workload
   - Monitor metrics for optimal performance

2. **Memory Management**
   - Configure cache size based on available memory
   - Enable disk cache for large workloads
   - Monitor memory usage through metrics

3. **Hardware Acceleration**
   - Enable CoreML on Apple Silicon
   - Use cache warming for frequently accessed embeddings
   - Configure thread counts based on CPU cores

4. **Production Settings**
   - Enable metrics collection
   - Configure appropriate timeouts
   - Set up monitoring dashboards
   - Enable cache warming

## Development

### Adding New Models

1. Add model to conversion script:
```python
SUPPORTED_MODELS = {
    "new-model-name": "org/model-name",
    ...
}
```

2. Convert model:
```bash
python scripts/convert_model.py --model new-model-name
```

3. Add model configuration:
```go
var ModelConfigs = map[string]ModelConfig{
    "new-model-name": {
        Name:       "new-model-name",
        Dimensions: 768,
        // Additional configuration...
    },
}
```

### Model Conversion Script
```python
[Your existing convert_model.py content with enhancements]
```

## Testing

### Requirements
All tests must be run through the provided Makefile due to required environment setup and native library dependencies. The Makefile handles:
- Dynamic library paths (ONNX Runtime, tokenizers)
- Platform-specific build tags
- Environment variables
- Test data setup

### Running Tests

```bash
# Run all tests (unit, integration, and benchmarks)
make test

# Run specific test suites
make test-unit          # Run unit tests only
make test-integration   # Run integration tests only
make test-benchmark     # Run performance benchmarks
```

### Test Organization
- `test/unit/`: Unit tests for individual components
- `test/integration/`: Integration tests for the full service
- `test/benchmark/`: Performance benchmarks
- `testdata/`: Test models and tokenizer files

### Common Issues
1. **Running tests directly with `go test`** will fail due to missing environment setup. Always use the Makefile targets.

2. **Missing test data**: If test data is missing, run:
```bash
make test-setup
```

3. **Library not found errors**: Ensure you've completed the installation steps:
```bash
make deps        # Install dependencies
make test-setup  # Set up test data and libraries
```

## Benchmarking

The library includes comprehensive benchmarks for:
- Single text embedding generation
- Batch processing
- Memory usage
- Metal acceleration performance

Run benchmarks:
```bash
go test -bench=. -benchmem ./pkg/embedding
```

Example benchmark output:
```
BenchmarkSingleEmbed-8              1000           1234567 ns/op          1234 B/op          12 allocs/op
BenchmarkBatchEmbed32-8             100           12345678 ns/op          5678 B/op          34 allocs/op
BenchmarkBatchEmbedMetal32-8        200            6789012 ns/op          5678 B/op          34 allocs/op
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
