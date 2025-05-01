#!/bin/bash

echo "Training the entity extractor BERT model..."
python scripts/python_scripts/train_entity_bert.py \
    --model-path "$ENTITY_BERT_MODEL_PATH" \
    --batch-size "$BATCH_SIZE" \
    --data-path "$ENTITY_BERT_DATA_PATH"
if [ $? -ne 0 ]; then
    echo "Error: train_entity_extractor.py failed. Exiting."
    exit 1
fi