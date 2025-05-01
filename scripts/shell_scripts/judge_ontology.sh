#!/bin/bash

echo "Running LLM judge for ontology..."
python scripts/python_scripts/judge_ontologies_with_llm.py \
    --data-source-name "$DATA_SOURCE_NAME" \
    --category-name "$CATEGORY_NAME" \
    --aspect-extraction-name "$ASPECT_EXTRACTION_NAME" \
    --ontology-extraction-name "$ONTOLOGY_EXTRACTION_NAME" \
    --save-to-db "true"
if [ $? -ne 0 ]; then
    echo "Error: judge_ontologies_with_llm.py failed. Exiting."
    exit 1
fi