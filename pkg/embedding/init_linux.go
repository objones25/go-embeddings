//go:build linux
// +build linux

package embedding

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/yalue/onnxruntime_go"
)

func init() {
	// Only run on Linux
	if runtime.GOOS != "linux" {
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
		libPath = filepath.Join(filepath.Dir(execPath), "libs", "libonnxruntime.so.1.20.0")
	}

	// Ensure the library exists
	if _, err := os.Stat(libPath); err != nil {
		panic(fmt.Sprintf("ONNX Runtime library not found at %s: %v", libPath, err))
	}

	// Set the library path for ONNX Runtime
	onnxruntime_go.SetSharedLibraryPath(libPath)

	// Update library path
	ldPath := os.Getenv("LD_LIBRARY_PATH")
	libDir := filepath.Dir(libPath)
	if ldPath == "" {
		ldPath = libDir
	} else {
		ldPath = libDir + ":" + ldPath
	}
	if err := os.Setenv("LD_LIBRARY_PATH", ldPath); err != nil {
		panic(fmt.Sprintf("failed to set LD_LIBRARY_PATH: %v", err))
	}
}
