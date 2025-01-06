
#!/bin/bash

# cleanup.sh
set -e  # Exit on any error

echo "🧹 Starting cleanup..."

# Function to safely remove directory
safe_remove() {
    local dir=$1
    if [ -d "$dir" ]; then
        echo "Removing $dir..."
        rm -rf "$dir"
    else
        echo "$dir not found, skipping..."
    fi
}

# Function to check if directory is empty
is_empty() {
    local dir=$1
    if [ -d "$dir" ] && [ "$(ls -A $dir)" ]; then
        return 1  # Directory exists and is not empty
    else
        return 0  # Directory doesn't exist or is empty
    fi
}

# Remove virtual environment
safe_remove "venv"

# Remove models directory if it exists
if [ -d "../models" ]; then
    echo "🗑️  Models directory found..."
    read -p "Do you want to remove all downloaded models? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        safe_remove "../models"
        echo "✅ Models removed successfully"
    else
        echo "⏭️  Skipping models removal"
    fi
fi

# Remove Python cache files
echo "🧹 Cleaning Python cache files..."
find .. -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
find .. -type f -name "*.pyc" -delete
find .. -type f -name "*.pyo" -delete
find .. -type f -name "*.pyd" -delete

# Remove any temporary files
echo "🧹 Cleaning temporary files..."
find .. -type f -name "*.log" -delete
find .. -type f -name "*.tmp" -delete

echo "✨ Cleanup complete!"