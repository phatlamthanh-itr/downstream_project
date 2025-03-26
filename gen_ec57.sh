#! /bin/sh



DEFAULT_SOURCE_DIR="val_prediction_atr"
DEFAULT_DEST_DIR="data/ecg_segmentation/mit-bih-arrhythmia-database-1.0.0"

SOURCE_DIR="${1:-$DEFAULT_SOURCE_DIR}"
DEST_DIR="${2:-$DEFAULT_DEST_DIR}"



# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist."
    exit 1
fi


# Copy .beat file
echo "Copying files from '$SOURCE_DIR' to '$DEST_DIR'..."
cp -rf "$SOURCE_DIR"/* "$DEST_DIR"/


cd EC57

if [ -d "output" ]; then
    echo "Existing output folder removed"
    rm -rf output
fi

python run.py



