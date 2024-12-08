#!/bin/bash

### Downsamples jsonl files for testing purposes ###

# Set the number of rows to sample
N="$1" 

# Define input and output directories
input_dir="data/datasets/prompted"
output_dir="data/datasets/prompted_sampled"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop over the JSONL files and create a randomly sampled version with only N rows
for file in "$input_dir"/*.jsonl; do
  filename=$(basename -- "$file")
  output_file="$output_dir/$filename"

  # Randomly shuffle lines and take the first N rows, then write to the new file
  shuf "$file" | head -n $N > "$output_file"

  echo "Randomly sampled $N rows from $filename and saved to $output_file"
done

echo "Random sampling complete."
