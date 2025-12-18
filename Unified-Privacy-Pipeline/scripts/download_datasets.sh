#!/bin/bash
# Automated dataset download script for Unified Privacy Pipeline

echo "========================================="
echo "Dataset Download Script"
echo "Unified Privacy Pipeline"
echo "========================================="
echo ""

# Change to project root
cd "$(dirname "$0")/.."

# Create dataset directories if they don't exist
mkdir -p datasets/face_recognition
mkdir -p datasets/health_prediction

# Function to download with progress
download_file() {
    local url=$1
    local output=$2
    local name=$3

    echo "Downloading $name..."
    if command -v wget &> /dev/null; then
        wget -q --show-progress "$url" -O "$output"
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar "$url" -o "$output"
    else
        echo "Error: Neither wget nor curl is installed"
        return 1
    fi

    if [ $? -eq 0 ]; then
        echo "[PASS] Downloaded $name"
        return 0
    else
        echo "[FAIL] Failed to download $name"
        return 1
    fi
}

# Download LFW Face Dataset
echo ""
echo "1. Face Recognition Dataset (LFW)"
echo "   Size: ~170MB"
echo "   This will take 1-2 minutes on fast connection..."
echo ""

if [ ! -d "datasets/face_recognition/lfw" ]; then
    cd datasets/face_recognition

    if download_file "https://vis-www.cs.umass.edu/lfw/lfw.tgz" "lfw.tgz" "LFW Dataset"; then
        echo "   Extracting..."
        tar -xzf lfw.tgz
        rm lfw.tgz
        echo "   [PASS] LFW dataset ready"

        # Count images
        num_images=$(find lfw -type f -name "*.jpg" | wc -l)
        echo "   Images: $num_images face images extracted"
    fi

    cd ../..
else
    echo "[SKIP] LFW dataset already exists"
fi

# Download Heart Disease Dataset
echo ""
echo "2. Heart Disease Dataset (UCI Cleveland)"
echo "   Size: <1MB"
echo ""

cd datasets/health_prediction

if [ ! -f "cleveland_heart.csv" ]; then
    if download_file "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data" \
                     "cleveland_heart.csv" \
                     "Heart Disease Dataset"; then
        echo "   [PASS] Heart disease dataset ready"
        lines=$(wc -l < cleveland_heart.csv)
        echo "   Samples: $lines patient records"
    fi
else
    echo "[SKIP] Heart disease dataset already exists"
fi

# Download Diabetes Dataset
echo ""
echo "3. Diabetes Dataset (UCI Pima Indians)"
echo "   Size: <1MB"
echo ""

if [ ! -f "diabetes.csv" ]; then
    if download_file "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv" \
                     "diabetes.csv" \
                     "Diabetes Dataset"; then
        echo "   [PASS] Diabetes dataset ready"
        lines=$(wc -l < diabetes.csv)
        echo "   Samples: $lines patient records"
    fi
else
    echo "[SKIP] Diabetes dataset already exists"
fi

cd ../..

# Summary
echo ""
echo "========================================="
echo "Download Summary"
echo "========================================="
echo ""

# Check what was downloaded
if [ -d "datasets/face_recognition/lfw" ]; then
    echo "[PASS] Face Recognition Dataset (LFW)"
else
    echo "[FAIL] Face Recognition Dataset - Not downloaded"
fi

if [ -f "datasets/health_prediction/cleveland_heart.csv" ]; then
    echo "[PASS] Heart Disease Dataset"
else
    echo "[FAIL] Heart Disease Dataset - Not downloaded"
fi

if [ -f "datasets/health_prediction/diabetes.csv" ]; then
    echo "[PASS] Diabetes Dataset"
else
    echo "[FAIL] Diabetes Dataset - Not downloaded"
fi

echo ""
echo "========================================="
echo "Next Steps"
echo "========================================="
echo ""
echo "1. Verify datasets:"
echo "   python3 scripts/verify_datasets.py"
echo ""
echo "2. Run demo with real data:"
echo "   python3 presentation_demo.py --use-real-data"
echo ""
echo "3. Train models:"
echo "   python3 scripts/train_model.py --task face_recognition"
echo "   python3 scripts/train_model.py --task health_prediction"
echo ""

# Calculate total size
total_size=$(du -sh datasets/ | cut -f1)
echo "Total dataset size: $total_size"
echo ""
echo "Download complete!"
