#!/bin/bash

echo "Running train_td_bert_on_semeval4.py..."
python scripts/python_scripts/train_td_bert_on_semeval4.py \
    --cache-dir "$CACHE_DIR" \
    --model-path "$SENTIMENT_BERT_MODEL_PATH"
if [ $? -ne 0 ]; then
    echo "Error: train_td_bert_on_semeval4.py failed. Exiting."
    exit 1
fi