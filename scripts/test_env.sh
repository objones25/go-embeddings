#!/bin/bash

# Get the absolute path to the project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Detect OS
OS=$(uname -s)
case "$OS" in
    "Darwin")
        LIB_NAME="libonnxruntime.1.20.0.dylib"
        LIB_PATH="${PROJECT_ROOT}/libs/${LIB_NAME}"
        export DYLD_LIBRARY_PATH="${PROJECT_ROOT}/libs:${DYLD_LIBRARY_PATH}"
        export DYLD_FALLBACK_LIBRARY_PATH="${PROJECT_ROOT}/libs:${DYLD_FALLBACK_LIBRARY_PATH}"
        ;;
    "Linux")
        LIB_NAME="libonnxruntime.so.1.20.0"
        LIB_PATH="${PROJECT_ROOT}/libs/${LIB_NAME}"
        export LD_LIBRARY_PATH="${PROJECT_ROOT}/libs:${LD_LIBRARY_PATH}"
        ;;
    *)
        echo "Unsupported operating system: $OS"
        exit 1
        ;;
esac

# Set up library paths
export ONNXRUNTIME_LIB_PATH="${LIB_PATH}"
export CGO_LDFLAGS="-L${PROJECT_ROOT}/libs -ltokenizers -lonnxruntime -Wl,-rpath,${PROJECT_ROOT}/libs"
export CGO_CFLAGS="-I${PROJECT_ROOT}/onnxruntime-osx-arm64-1.20.0/include"

# Create symlinks if needed
if [ "$OS" = "Darwin" ]; then
    cd "${PROJECT_ROOT}/libs" || exit 1
    ln -sf "${LIB_NAME}" "libonnxruntime.dylib"
    ln -sf "${LIB_NAME}" "onnxruntime.so"
    ln -sf "libtokenizers.a" "libtokenizers.dylib"
    
    # Fix RPATH for dynamic libraries
    install_name_tool -id "@rpath/libonnxruntime.1.20.0.dylib" "${LIB_NAME}"
    install_name_tool -add_rpath "${PROJECT_ROOT}/libs" "${LIB_NAME}"
    cd - || exit 1
fi

# Print environment for debugging
echo "Environment setup complete:"
echo "OS: $OS"
echo "ONNXRUNTIME_LIB_PATH=$ONNXRUNTIME_LIB_PATH"
echo "CGO_LDFLAGS=$CGO_LDFLAGS"
echo "CGO_CFLAGS=$CGO_CFLAGS"
if [ "$OS" = "Darwin" ]; then
    echo "DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH"
    echo "DYLD_FALLBACK_LIBRARY_PATH=$DYLD_FALLBACK_LIBRARY_PATH"
else
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
fi

# List library files
echo "Library files in ${PROJECT_ROOT}/libs:"
ls -l "${PROJECT_ROOT}/libs"

# Print RPATH information for dynamic libraries
if [ "$OS" = "Darwin" ]; then
    echo "RPATH information for ${LIB_NAME}:"
    otool -l "${LIB_PATH}" | grep -A2 LC_RPATH
fi 