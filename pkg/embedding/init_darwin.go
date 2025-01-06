//go:build darwin && coreml
// +build darwin,coreml

package embedding

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/yalue/onnxruntime_go"
)

func init() {
	// Only run on Darwin (macOS)
	if runtime.GOOS != "darwin" {
		return
	}

	// Get the ONNX Runtime library path from environment
	libPath := os.Getenv("ONNXRUNTIME_LIB_PATH")
	if libPath == "" {
		// Default to libs directory relative to binary
		execPath, err := os.Executable()
		if err != nil {
			panic(fmt.Sprintf("failed to get executable path: %v", err))
		}
		libPath = filepath.Join(filepath.Dir(execPath), "libs", "libonnxruntime.1.20.0.dylib")
	}

	// Ensure the library exists
	if _, err := os.Stat(libPath); err != nil {
		panic(fmt.Sprintf("ONNX Runtime library not found at %s: %v", libPath, err))
	}

	// Set the library path for ONNX Runtime
	onnxruntime_go.SetSharedLibraryPath(libPath)

	// Update dynamic library path
	dyldPath := os.Getenv("DYLD_LIBRARY_PATH")
	libDir := filepath.Dir(libPath)
	if dyldPath == "" {
		dyldPath = libDir
	} else {
		dyldPath = libDir + ":" + dyldPath
	}
	if err := os.Setenv("DYLD_LIBRARY_PATH", dyldPath); err != nil {
		panic(fmt.Sprintf("failed to set DYLD_LIBRARY_PATH: %v", err))
	}
}
