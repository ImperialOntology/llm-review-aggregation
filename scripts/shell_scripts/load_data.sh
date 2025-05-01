#!/bin/bash

echo "Running load_data_into_db.py..."
python scripts/python_scripts/load_data_into_db.py \
    --cache-dir "$CACHE_DIR" \
    --data-source-name "$DATA_SOURCE_NAME" \
    --data-source-url "$DATA_SOURCE_URL" \
    --category-name "$CATEGORY_NAME" \
    --reviews "$NUMBER_OF_REVIEWS" \
    --data-source "$DATA_SOURCE"
if [ $? -ne 0 ]; then
    echo "Error: load_data_into_db.py failed. Exiting."
    exit 1
fi