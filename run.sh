#!/bin/bash
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1:gpu_type=L40S
#PBS -N llm-review-aggregation-job


cd $PBS_O_WORKDIR

# Setup env variables and the python venv
. scripts/shell_scripts/setup.sh

# Create a subdirectory in $PBS_O_WORKDIR
SUBDIR_NAME="llm_review_aggregation_output"  # Name of the subdirectory
SUBDIR_PATH="$PBS_O_WORKDIR/$SUBDIR_NAME"    # Full path to the subdirectory
CACHE_DIR="$SUBDIR_PATH/cache"               # Subdirectory for caching

echo "Creating subdirectory: $SUBDIR_PATH"
mkdir -p "$SUBDIR_PATH" "$CACHE_DIR"

# Check if the subdirectory was created successfully
if [ -d "$SUBDIR_PATH" ]; then
    echo "Subdirectory created successfully."
else
    echo "Failed to create subdirectory. Exiting."
    exit 1
fi

SENTIMENT_BERT_MODEL_PATH="$CACHE_DIR/SentimentBertModel.model"
ENTITY_BERT_MODEL_PATH="$CACHE_DIR/bert_entity_extractor.pt"
RELATION_BERT_MODEL_PATH="$CACHE_DIR/bert_relation_extractor.pt"

ENTITY_BERT_DATA_PATH="$PBS_O_WORKDIR/bert-train-data/term_extraction_datasets"
RELATION_BERT_DATA_PATH="$PBS_O_WORKDIR/bert-train-data/relation_extraction_datasets"

# make sure the BERT_OR_LLM is set to "llm" or "bert"
if [ "$BERT_OR_LLM" != "llm" ] && [ "$BERT_OR_LLM" != "bert" ]; then
    echo "Error: BERT_OR_LLM must be set to 'llm' or 'bert'. Exiting."
    exit 1
fi


# 1. Load data into the database
. scripts/shell_scripts/load_data.sh

# 2. Train the entity extractor BERT model if necessary
# aka if method is set to BERT and if the model does not exist or if retrain is set to true
if [ "$BERT_OR_LLM" = "bert" ]; then  
    if [ ! -f "$ENTITY_BERT_MODEL_PATH" ] || [ "$RETRAIN_ENTITY_BERT" = "true" ]; then
        . scripts/shell_scripts/train_entity_bert.sh
    else
        echo "Entity extractor BERT model already exists. Skipping training."
    fi
fi

# 3. Run aspect extraction
. scripts/shell_scripts/extract_aspects.sh


# 4. Train relation extractor BERT model if necessary
# aka if method is set to BERT and if the model does not exist or if retrain is set to true
if [ "$BERT_OR_LLM" = "bert" ]; then  
    if [ ! -f "$RELATION_BERT_MODEL_PATH" ] || [ "$RETRAIN_RELATION_BERT" = "true" ]; then
        . scripts/shell_scripts/train_relation_bert.sh
    else
        echo "Relation extractor BERT model already exists. Skipping training."
    fi
fi

# 5. Run ontology extraction
. scripts/shell_scripts/extract_ontology.sh

# 6. Train sentiment BERT if necessary
# aka if the model does not exist or if retrain is set to true
if [ ! -f "$SENTIMENT_BERT_MODEL_PATH" ] || [ "$RETRAIN_SENTIMENT_BERT" = "true" ]; then
    . scripts/shell_scripts/train_sentiment_bert.sh
else
    echo "Sentiment BERT model already exists. Skipping training."
fi

# 7. Run argumentation analysis
. scripts/shell_scripts/run_argumentation.sh

# 8. Run LLM judge for aspects if RUN_JUDGE is configured to true
. scripts/shell_scripts/judge_aspects.sh


# 9. Run LLM judge for ontology if RUN_JUDGE is configured to true
. scripts/shell_scripts/judge_ontology.sh

echo "All scripts have been run successfully."
