#!/bin/bash -l 
#COBALT -t 6:00:00 -n 1 -q gpu_h100 --jobname llama_2_7b_f16

module load cuda/12.3.0

cd /vast/users/sraskar/h100/llm_research/llama.cpp/build/bin

for batch_size in 1 16 32 64; do
    for input_length in 128 256 512 1024 2048; do
        for output_length in 128 256 512 1024 2048; do
            for CVD in "0" "0,1" "0,1,2,3";do
                CUDA_VISIBLE_DEVICES=$CVD ./llama-bench -m /vast/users/sraskar/model_weights/GGUF_weights/llama_2_7b_f16.gguf -p $input_length -n $output_length -pg $input_length,$output_length -b $batch_size -r 1 -o csv
            done
        done
    done
done