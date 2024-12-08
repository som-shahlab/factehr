#!/bin/bash

sizes=(4000 5000)

base_cmd="python tune_eval.py --use_peft --peft_method lora --dataset ehr_dataset --quantization --use_fp16 --model_name /share/pi/nigam/pretrained/Meta-Llama-3-8B-Instruct --context_length=8192 --batch_size_training=2 --num_epochs=5"

for size in "${sizes[@]}"; do
    echo "Processing dataset size: $size"

    cmd="$base_cmd --ehr_dataset.data_path ../data/ehr_data_${size}.json --output_dir /share/pi/nigam/projects/synth-instruct/models/models_ehr_${size}"

    echo "Running command: $cmd"
    eval $cmd

    echo "Completed processing for size $size"
    echo "----------------------------------------"
done

echo "All processing completed."
