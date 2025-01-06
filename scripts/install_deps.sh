#!/bin/bash

# Exit on any error
set -e

# Create libs directory if it doesn't exist
mkdir -p libs

# Function to check if file exists
check_file() {
    local file=$1
    if [ -f "$file" ]; then
        echo "✅ Found $file"
        return 0
    else
        return 1
    fi
}

# Check for existing files first
if check_file "libs/libonnxruntime.1.20.0.dylib" && check_file "libs/libtokenizers.a"; then
    echo "✅ All required libraries are present"
    exit 0
fi

# Download and install tokenizers if needed
if ! check_file "libs/libtokenizers.a"; then
    echo "📥 Downloading tokenizers..."
    curl -L -o libs/tokenizers.tar.gz https://github.com/daulet/tokenizers/releases/download/v1.20.2/libtokenizers.darwin-arm64.tar.gz
    tar -xzf libs/tokenizers.tar.gz -C libs/
    rm libs/tokenizers.tar.gz
fi

# Download ONNX Runtime if needed
if ! check_file "libs/libonnxruntime.1.20.0.dylib"; then
    echo "📥 Downloading ONNX Runtime..."
    curl -L -o libs/onnxruntime.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-osx-arm64-1.20.0.tgz
    tar -xzf libs/onnxruntime.tgz -C libs/ onnxruntime-osx-arm64-1.20.0/lib/libonnxruntime.1.20.0.dylib
    mv libs/onnxruntime-osx-arm64-1.20.0/lib/libonnxruntime.1.20.0.dylib libs/
    rm -rf libs/onnxruntime.tgz libs/onnxruntime-osx-arm64-1.20.0
fi

# Set library path
export LIBRARY_PATH=$LIBRARY_PATH:$(pwd)/libs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/libs 

echo "✅ Dependencies installation complete!" 