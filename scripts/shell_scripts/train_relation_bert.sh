#!/bin/bash

echo "Training the relation extractor BERT model..."
python scripts/python_scripts/train_relation_bert.py \
    --model-path "$RELATION_BERT_MODEL_PATH" \
    --batch-size "$BATCH_SIZE" \
    --data-path "$RELATION_BERT_DATA_PATH"
if [ $? -ne 0 ]; then
    echo "Error: train_relation_extractor.py failed. Exiting."
    exit 1
fi