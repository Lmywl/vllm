#!/bin/bash

# Requirement: 2x GPUs.

# Model: meta-llama/Meta-Llama-3.1-8B-Instruct
# Query: 1024 input tokens, 6 output tokens, QPS 2/4/6/8, 100 requests
# Resource: 2x GPU
# Approaches:
# 2. Chunked prefill: 2 vllm instance with tp=4, equivalent to 1 tp=4 instance with QPS 4
# 3. Disaggregated prefill: 1 prefilling instance and 1 decoding instance
# Prefilling instance: max_output_token=1
# Decoding instance: force the input tokens be the same across requests to bypass prefilling

set -ex
model="/root/paddlejob/workspace/env_run/liumengyuan/models/qwen2.5_1.5B_Instruct" 
kill_gpu_processes() {
  # kill all processes on GPU.
  pgrep pt_main_thread | xargs -r kill -9
  pgrep python3 | xargs -r kill -9
  for port in 8787 8300 8200; do lsof -t -i:$port | xargs -r kill -9; done
  sleep 1
}

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}


launch_disagg_prefill() {
  # model="/root/paddlejob/workspace/env_run/liumengyuan/models/qwen2.5_1.5B_Instruct" 
  # disagg prefill
  WORKER_TYPE=Prefill CUDA_VISIBLE_DEVICES=2 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8300 \
    --max-model-len 10000 \
    --gpu-memory-utilization 0.6 \
    --kv-transfer-config \
    '{"kv_connector":"SharedStorageConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":5e9}' &

  WORKER_TYPE=Decode CUDA_VISIBLE_DEVICES=3 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8200 \
    --max-model-len 10000 \
    --gpu-memory-utilization 0.6 \
    --kv-transfer-config \
    '{"kv_connector":"SharedStorageConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":5e9}' &

  wait_for_server 8300
  wait_for_server 8200
  python3 disagg_prefill_proxy_server.py &
  sleep 1
}


main() {
  launch_disagg_prefill
}


main "$@"
