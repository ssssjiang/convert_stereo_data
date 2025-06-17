#!/bin/bash
# This script finds all data.csv files in the specified root directory (or current directory if not provided)
# and its subdirectories, and deletes lines 2 to 12 from each file.

ROOT_DIR="${1:-.}"

if [ ! -d "$ROOT_DIR" ]; then
    echo "Error: Directory '$ROOT_DIR' not found."
    echo "Usage: $0 [root_directory]"
    exit 1
fi

echo "Searching for data.csv files in '$ROOT_DIR'..."

find "$ROOT_DIR" -type f -name "data.csv" -print0 | while IFS= read -r -d $'\0' file; do
    # Check if the file has more than 1 line to avoid errors with empty or single-line files
    if [ "$(wc -l < "$file")" -gt 1 ]; then
        sed -i '2,12d' "$file"
        echo "Processed $file"
    else
        echo "Skipping $file (has 1 or 0 lines)"
    fi
done

echo "Done." 