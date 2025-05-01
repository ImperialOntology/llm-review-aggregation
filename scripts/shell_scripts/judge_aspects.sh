#!/bin/bash

echo "Running LLM judge for aspects..."
python scripts/python_scripts/judge_aspects_with_llm.py \
    --data-source-name "$DATA_SOURCE_NAME" \
    --category-name "$CATEGORY_NAME" \
    --aspect-extraction-name "$ASPECT_EXTRACTION_NAME" \
    --save-to-db "true"
if [ $? -ne 0 ]; then
    echo "Error: judge_aspects_with_llm.py failed. Exiting."
    exit 1
fi