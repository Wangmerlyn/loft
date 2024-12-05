BASE_DIR=datasets
PROJECT_ID=your-gcp-project-id
MODEL_PATH=/mnt/longcontext/models/siyuan/llama3/llama-3.1-8B
# get basename
MODEL_NAME=$(basename $MODEL_PATH)
DATASET_LIST=("arguana" "fever" "fiqa" "msmarco" "nq" "quora" "scifact" "webis_touche2020" "topiocqa" "hotpotqa" "musique" "qampari" "quest")

for DATASET in "${DATASET_LIST[@]}"; do
    echo "Processing dataset: ${DATASET}"
    # export as environment variable
    export DATASET    
    # Run inference
    python run_inference.py \
        --prompt_prefix_path ${BASE_DIR}/prompts/retrieval_128k/retrieval_${DATASET}_128k.txt \
        --data_dir ${BASE_DIR}/data/retrieval/${DATASET}/128k \
        --split dev \
        --context_length 128k \
        --output_path ${BASE_DIR}/outputs/${MODEL_NAME}/retrieval/${DATASET}/128k/predictions.jsonl \
        --project_id ${PROJECT_ID} \
        --model_url_or_name ${MODEL_PATH}

    # Run evaluation
    python run_evaluation.py \
        --answer_file_path ${BASE_DIR}/data/retrieval/${DATASET}/128k/dev_queries.jsonl \
        --pred_file_path ${BASE_DIR}/outputs/${MODEL_NAME}/retrieval/${DATASET}/128k/predictions.jsonl \
        --task_type retrieval
done
