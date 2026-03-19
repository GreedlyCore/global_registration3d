#!/bin/bash
set -e

# Download 3 sequences of Mulran dataset from google drive using gdown

BASE_DIR="$(pwd)/MulRan"
mkdir -p "$BASE_DIR"

download_file() {
    gdown "$1" --fuzzy --continue --output "$2/"
}

extract_ouster() {
    local archive="$1"
    local target_dir="$2"
    local temp_dir=$(mktemp -d)
    
    echo "Extracting $(basename "$archive")..."
    
    # Check if it's gzip compressed or just tar
    if file "$archive" | grep -q "gzip compressed"; then
        tar -xzf "$archive" -C "$temp_dir"
    else
        tar -xf "$archive" -C "$temp_dir"
    fi
    
    # Handle potential nested directory structure
    if [ $(ls -1 "$temp_dir" | wc -l) -eq 1 ] && [ -d "$temp_dir"/* ]; then
        # Single directory found inside - move its contents
        mv "$temp_dir"/*/* "$target_dir/Ouster/" 2>/dev/null || true
        # Also try to move hidden files if any
        mv "$temp_dir"/*/.[!.]* "$target_dir/Ouster/" 2>/dev/null || true
    else
        # Multiple files/folders - move everything
        mv "$temp_dir"/* "$target_dir/Ouster/" 2>/dev/null || true
        mv "$temp_dir"/.[!.]* "$target_dir/Ouster/" 2>/dev/null || true
    fi
    
    # Clean up
    rm -rf "$temp_dir"
    rm "$archive"
    echo "Done extracting to $target_dir/Ouster/"
}

# DCC02
echo "Processing DCC02..."
mkdir -p "$BASE_DIR/DCC02/Ouster"
download_file "https://drive.google.com/file/d/1GRLG1q-aWB55U-FqF5ljSCi2p1sBEeZh/view?usp=drive_link" "$BASE_DIR/DCC02"
download_file "https://drive.google.com/file/d/1vxGEnMIivfTGI4GYfsXwKJw9LIuLH78F/view?usp=drive_link" "$BASE_DIR/DCC02"
download_file "https://drive.google.com/file/d/1V_mV2JAytseHEl8dhmQY0C1E3SgWVwNR/view?usp=drive_link" "$BASE_DIR/DCC02"
extract_ouster "$BASE_DIR/DCC02/Ouster.tar.gz" "$BASE_DIR/DCC02"

# KAIST02
echo "Processing KAIST02..."
mkdir -p "$BASE_DIR/KAIST02/Ouster"
download_file "https://drive.google.com/file/d/1jjqCronWNblAkIvP5vQG_y8YSfa0zpuU/view?usp=drive_link" "$BASE_DIR/KAIST02"
download_file "https://drive.google.com/file/d/1I-4Vpfs1TzpeRRXIMPoBDRvyULKbrp6v/view?usp=drive_link" "$BASE_DIR/KAIST02"
download_file "https://drive.google.com/file/d/1UuiblPMVVoghGmX-AY7P-k6ZkggbBIal/view?usp=drive_link" "$BASE_DIR/KAIST02"
extract_ouster "$BASE_DIR/KAIST02/Ouster.tar.gz" "$BASE_DIR/KAIST02"

# RIVERSIDE02
echo "Processing RIVERSIDE02..."
mkdir -p "$BASE_DIR/RIVERSIDE02/Ouster"
download_file "https://drive.google.com/file/d/17DQqtomY4bAnEpXq8rXjvWMyTWGYdpfT/view?usp=drive_link" "$BASE_DIR/RIVERSIDE02"
download_file "https://drive.google.com/file/d/1Re5pF-dKTG6UnRklsG1vSdnfBBUm9FO7/view?usp=drive_link" "$BASE_DIR/RIVERSIDE02"
download_file "https://drive.google.com/file/d/1xOnOUeweckFpR-Wqff6qYJ8GiCkLJ2wV/view?usp=drive_link" "$BASE_DIR/RIVERSIDE02"
extract_ouster "$BASE_DIR/RIVERSIDE02/Ouster.tar.gz" "$BASE_DIR/RIVERSIDE02"

echo "All downloads and extractions complete!"
echo "Final directory structure:"
tree -L 3 "$BASE_DIR" || ls -R "$BASE_DIR"