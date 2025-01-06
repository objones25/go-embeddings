# Go parameters
GOCMD=go
GOBUILD=$(GOCMD) build
GOTEST=$(GOCMD) test
GOGET=$(GOCMD) get
GOMOD=$(GOCMD) mod
GOFMT=$(GOCMD) fmt
BINARY_NAME=embeddings

# Python parameters
PYTHON=python3
VENV=venv
PIP=$(VENV)/bin/pip

# Directories
BUILD_DIR=build
MODELS_DIR=models
SCRIPTS_DIR=scripts
TEST_DATA_DIR=testdata

# Library paths and flags
LIBS_DIR=$(shell pwd)/libs
CGO_FLAGS=-L$(LIBS_DIR)
CGO_LDFLAGS=$(CGO_FLAGS)
export CGO_LDFLAGS
export DYLD_LIBRARY_PATH=$(LIBS_DIR)
export LD_LIBRARY_PATH=$(LIBS_DIR)
export ONNXRUNTIME_LIB_PATH=$(LIBS_DIR)/libonnxruntime.1.20.0.dylib

# Test data URLs
TEST_MODEL_URL=https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.onnx
TEST_TOKENIZER_URL=https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json

.PHONY: all build test test-unit test-integration test-benchmark clean fmt lint deps setup test-setup

all: clean deps fmt lint test build

help: ## Display this help message
	@echo "Usage:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

clean: ## Clean build directory and Python cache
	rm -rf $(BUILD_DIR)
	rm -rf **/__pycache__
	rm -f *.out
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

deps: go-deps python-deps ## Install all dependencies

go-deps: ## Install Go dependencies
	$(GOMOD) download
	$(GOMOD) tidy
	$(GOMOD) verify

python-deps: ## Install Python dependencies
	test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r $(SCRIPTS_DIR)/requirements.txt

fmt: ## Format Go code
	$(GOFMT) ./...

vet: ## Vet Go code
	$(GOCMD) vet ./...

lint: ## Lint Go code
	golangci-lint run ./...

build: ## Build the application
	mkdir -p $(BUILD_DIR)
	CGO_LDFLAGS="$(CGO_FLAGS)" $(GOBUILD) -o $(BUILD_DIR)/$(BINARY_NAME) ./cmd/embedserver

test: test-setup test-unit test-integration test-benchmark

test-unit: test-setup ## Run unit tests
	source $(SCRIPTS_DIR)/test_env.sh && \
	GOOS=$(shell go env GOOS) $(GOTEST) -v -race -tags="$(shell go env GOOS),coreml" ./test/unit/...

test-integration: test-setup ## Run integration tests
	source $(SCRIPTS_DIR)/test_env.sh && \
	GOOS=$(shell go env GOOS) $(GOTEST) -v -race -tags="$(shell go env GOOS),coreml" ./test/integration/...

test-benchmark: test-setup ## Run benchmark tests
	source $(SCRIPTS_DIR)/test_env.sh && \
	GOOS=$(shell go env GOOS) $(GOTEST) -v -race -tags="$(shell go env GOOS),coreml" -bench=. -benchmem ./test/benchmark/...

test-setup: ## Setup test data and dependencies
	mkdir -p $(TEST_DATA_DIR)
	@echo "Downloading test model..."
	@if [ ! -f "$(TEST_DATA_DIR)/model.onnx" ]; then \
		curl -L -o $(TEST_DATA_DIR)/model.onnx $(TEST_MODEL_URL); \
	fi
	@echo "Downloading test tokenizer..."
	@if [ ! -f "$(TEST_DATA_DIR)/tokenizer.json" ]; then \
		curl -L -o $(TEST_DATA_DIR)/tokenizer.json $(TEST_TOKENIZER_URL); \
	fi
	@echo "Ensuring library paths..."
	@if [ ! -f "$(LIBS_DIR)/libonnxruntime.1.20.0.dylib" ]; then \
		echo "Error: ONNX Runtime library not found in libs/"; \
		exit 1; \
	fi
	@if [ ! -f "$(LIBS_DIR)/libtokenizers.a" ]; then \
		echo "Error: Tokenizers library not found in libs/"; \
		exit 1; \
	fi
	@echo "Creating symlink for ONNX Runtime library..."
	@cd $(LIBS_DIR) && ln -sf libonnxruntime.1.20.0.dylib libonnxruntime.dylib && ln -sf libonnxruntime.1.20.0.dylib onnxruntime.so

bench: ## Run benchmarks
	CGO_LDFLAGS="$(CGO_FLAGS)" $(GOTEST) -bench=. -benchmem ./...

run: build ## Run the application
	./$(BUILD_DIR)/$(BINARY_NAME)

models: download-models convert-models ## Download and convert all models

download-models: python-deps ## Download models
	@echo "Downloading models..."
	./$(SCRIPTS_DIR)/setup_and_run.sh

convert-models: python-deps ## Convert models to ONNX format
	@echo "Converting models to ONNX format..."
	$(PYTHON) $(SCRIPTS_DIR)/convert_model.py --output-dir $(MODELS_DIR)

docker-build: ## Build Docker image
	docker-compose build

docker-up: ## Start Docker containers
	docker-compose up -d

docker-down: ## Stop Docker containers
	docker-compose down

docker-logs: ## View Docker logs
	docker-compose logs -f

setup: ## Initial project setup
	mkdir -p $(MODELS_DIR)
	$(MAKE) deps
	$(MAKE) download-models

dev: ## Run development environment
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Variables for versioning
VERSION ?= $(shell git describe --tags --always --dirty)
COMMIT ?= $(shell git rev-parse --short HEAD)
BUILD_TIME ?= $(shell date -u '+%Y-%m-%d_%H:%M:%S')

# Download test model if it doesn't exist
testdata/model.onnx:
	mkdir -p testdata
	curl -L -o testdata/model.onnx https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.onnx

setup: deps testdata/model.onnx