#!/bin/bash
# This file demonstrates the example usage of disaggregated prefilling
# We will launch 2 vllm instances (1 for prefill and 1 for decode),
# and then transfer the KV cache between them.

set -xe

echo "🚧🚧 Warning: The usage of disaggregated prefill is experimental and subject to change 🚧🚧"
sleep 1

# meta-llama/Meta-Llama-3.1-8B-Instruct or deepseek-ai/DeepSeek-V2-Lite
# MODEL_NAME=${HF_MODEL_NAME:-meta-llama/Meta-Llama-3.1-8B-Instruct}
MODEL_NAME=/root/paddlejob/workspace/env_run/liumengyuan/models/qwen2.5_1.5B_Instruct

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'cleanup' INT

# Cleanup function
cleanup() {
    echo "Caught Ctrl+C, cleaning up..."
    # Cleanup commands
    pgrep python3 | xargs kill -9
    pgrep "vllm serve" | xargs kill -9
    echo "Cleanup complete. Exiting."
    exit 0
}

export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

# install quart first -- required for disagg prefill proxy serve
if python3 -c "import quart" &> /dev/null; then
    echo "Quart is already installed."
else
    echo "Quart is not installed. Installing..."
    python3 -m pip install quart
fi 

# a function that waits vLLM server to start
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}


# You can also adjust --kv-ip and --kv-port for distributed inference.

# prefilling instance, which is the KV producer
CUDA_VISIBLE_DEVICES=2 vllm serve $MODEL_NAME \
    --port 8300 \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"SharedStorageConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}' &

# decoding instance, which is the KV consumer
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL_NAME \
    --port 8200 \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"SharedStorageConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}' &

# wait until prefill and decode instances are ready
wait_for_server 8300
wait_for_server 8200

# launch a proxy server that opens the service at port 8787
# the workflow of this proxy:
# - send the request to prefill vLLM instance (port 8100), change max_tokens 
#   to 1
# - after the prefill vLLM finishes prefill, send the request to decode vLLM 
#   instance
# NOTE: the usage of this API is subject to change --- in the future we will 
# introduce "vllm connect" to connect between prefill and decode instances
python3 ../../benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py &
sleep 5

# serve two example requests
output1=$(curl -X POST -s http://localhost:8787/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "'"$MODEL_NAME"'",
"prompt": "San Francisco is a",
"max_tokens": 10,
"temperature": 0
}')

output2=$(curl -X POST -s http://localhost:8787/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "'"$MODEL_NAME"'",
"prompt": "Santa Clara is a",
"max_tokens": 10,
"temperature": 0
}')


echo ""
sleep 1

## clean proxy_server and vllm instances
pgrep python3 | xargs kill -9  # delete proxy_server
pkill -f "vllm serve"  # delete vllm instances

# Print the outputs of the curl requests
echo ""
echo "Output of first request: $output1"
echo "Output of second request: $output2"

echo "🎉🎉 Successfully finished 2 test requests! 🎉🎉"
echo ""


