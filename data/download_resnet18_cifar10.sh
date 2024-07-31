#!/bin/bash

# Define the list of URLs and target directory
URLS=(
    "https://zenodo.org/records/6977382/files/cifar100_resnet18_epoch60.zip"
)
TARGET_DIR="."

# Create the target directory if it doesn't exist
mkdir -p ${TARGET_DIR}

# Loop through each URL
for URL in "${URLS[@]}"; do
    # Get the base name of the file from the URL
    FILE_NAME=$(basename ${URL})
    
    # Define the output file path
    OUTPUT_FILE="${TARGET_DIR}/${FILE_NAME}"
    
    # Download the zip file
    curl -L ${URL} -o ${OUTPUT_FILE}
    
    # Unzip the downloaded file
    unzip ${OUTPUT_FILE} -d ${TARGET_DIR}
    
    # Optionally, remove the zip file after extraction
    rm ${OUTPUT_FILE}
    
    echo "Downloaded and extracted ${FILE_NAME}."
done

echo "All downloads and extractions complete."
