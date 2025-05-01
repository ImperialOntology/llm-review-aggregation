#!/bin/bash

echo "Running analyse_and_save_argumentation.py..."
python scripts/python_scripts/analyse_and_save_argumentation.py \
    --data-source-name "$DATA_SOURCE_NAME" \
    --category-name "$CATEGORY_NAME" \
    --aspect-extraction-name "$ASPECT_EXTRACTION_NAME" \
    --ontology-extraction-name "$ONTOLOGY_EXTRACTION_NAME" \
    --name "$ARGUMENT_ANALYSIS_NAME" \
    --method "$ARGUMENT_ANALYSIS_METHOD" \
    --description "$ARGUMENT_ANALYSIS_DESCRIPTION" \
    --ba-model-path "$SENTIMENT_BERT_MODEL_PATH"
if [ $? -ne 0 ]; then
    echo "Error: analyse_and_save_argumentation.py failed. Exiting."
    exit 1
fi