#!/bin/bash

# Define the URL and target directory
URL="https://zenodo.org/records/13144018/files/cifar10_cnn_sample_ep21-25.zip"

TARGET_DIR="."

# Create the target directory if it doesn't exist
mkdir -p ${TARGET_DIR}

# Define the output file path
OUTPUT_FILE="${TARGET_DIR}/cifar10_cnn_sample_ep21-25.zip"

# Download the zip file
curl -L ${URL} -o ${OUTPUT_FILE}

# Unzip the downloaded file
unzip ${OUTPUT_FILE} -d ${TARGET_DIR}

# Optionally, remove the zip file after extraction
rm ${OUTPUT_FILE}

echo "Download and extraction complete."
