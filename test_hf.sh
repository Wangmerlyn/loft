#!/bin/bash

# Usage: ./script.sh MODEL_PATH TEST_LENGTH

# Check for input arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 MODEL_PATH TEST_LENGTH"
    exit 1
fi

MODEL_PATH=$1
TEST_LENGTH=$2

BASE_DIR=datasets
PROJECT_ID=your-gcp-project-id
SPLIT=test

# Get basename
MODEL_NAME=$(basename $MODEL_PATH)
DATASET_LIST=("arguana" "fever" "fiqa" "msmarco" "nq" "quora" "scifact" "webis_touche2020" "topiocqa" "hotpotqa" "musique" "qampari" "quest")

for DATASET in "${DATASET_LIST[@]}"; do
    echo "Processing dataset: ${DATASET}"
    # Export as environment variable
    export DATASET    
    # Run inference
    python run_inference.py \
        --prompt_prefix_path ${BASE_DIR}/prompts/retrieval_128k/retrieval_${DATASET}_128k.txt \
        --data_dir ${BASE_DIR}/data/retrieval/${DATASET}/${TEST_LENGTH} \
        --split ${SPLIT} \
        --context_length ${TEST_LENGTH} \
        --output_path ${BASE_DIR}/outputs/${MODEL_NAME}/retrieval/${DATASET}/${TEST_LENGTH}/predictions.jsonl \
        --project_id ${PROJECT_ID} \
        --model_url_or_name ${MODEL_PATH}

    # Run evaluation
    python run_evaluation.py \
        --answer_file_path ${BASE_DIR}/data/retrieval/${DATASET}/${TEST_LENGTH}/${SPLIT}_queries.jsonl \
        --pred_file_path ${BASE_DIR}/outputs/${MODEL_NAME}/retrieval/${DATASET}/${TEST_LENGTH}/predictions.jsonl \
        --task_type retrieval
done
