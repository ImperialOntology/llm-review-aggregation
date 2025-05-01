#!/bin/bash

echo "Running extract_and_save_aspects.py..."
python scripts/python_scripts/extract_and_save_aspects.py \
    --data-source-name "$DATA_SOURCE_NAME" \
    --category-name "$CATEGORY_NAME" \
    --cache-dir "$CACHE_DIR" \
    --name "$ASPECT_EXTRACTION_NAME" \
    --method "$ASPECT_EXTRACTION_METHOD" \
    --description "$ASPECT_EXTRACTION_DESCRIPTION" \
    --batch-size "$BATCH_SIZE" \
    --bert-or-llm "$BERT_OR_LLM" \
    --bert-model-path "$ENTITY_BERT_MODEL_PATH" \
    --njobs "$NJOBS"
if [ $? -ne 0 ]; then
    echo "Error: extract_and_save_aspects.py failed. Exiting."
    exit 1
fi