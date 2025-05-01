#!/bin/bash

echo "Running extract_and_save_ontology.py..."
python scripts/python_scripts/extract_and_save_ontology.py \
    --data-source-name "$DATA_SOURCE_NAME" \
    --category-name "$CATEGORY_NAME" \
    --aspect-extraction-name "$ASPECT_EXTRACTION_NAME" \
    --name "$ONTOLOGY_EXTRACTION_NAME" \
    --method "$ONTOLOGY_EXTRACTION_METHOD" \
    --description "$ONTOLOGY_EXTRACTION_DESCRIPTION" \
    --cache-dir "$CACHE_DIR" \
    --top-k-aspects-to-keep "$TOP_K_ASPECTS_TO_KEEP" \
    --batch-size "$BATCH_SIZE" \
    --bert-or-llm "$BERT_OR_LLM" \
    --bert-model-path "$RELATION_BERT_MODEL_PATH" \
    --njobs "$NJOBS"
if [ $? -ne 0 ]; then
    echo "Error: extract_and_save_ontology.py failed. Exiting."
    exit 1
fi