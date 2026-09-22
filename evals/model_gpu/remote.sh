#!/usr/bin/env bash
set -u
cd /home/mingtian/typellm-model-gpu
mkdir -p output hf-cache
PY=/home/mingtian/typear-venv/bin/python
$PY -m unittest discover -s tests >output/unit-tests.log 2>&1 || exit 1
sudo docker stop sglang-qwen38 >/dev/null
for spec in 'minicpm5 openbmb/MiniCPM5-1B' 'ling inclusionAI/Ling-mini-2.0' 'ring inclusionAI/Ring-mini-2.0'; do
    read -r label model <<< "$spec"
    name="typellm-protocol-${label}-20260922"
    mkdir -p "output/$label" "hf-cache/$label"
    echo "START $model $(date -u +%FT%TZ)"
    sudo docker run -d --name "$name" --gpus all --network host --shm-size 16g \
        -v "$PWD/hf-cache/$label:/root/.cache/huggingface" \
        -v /home/mingtian/.cache/huggingface:/existing-hf:ro \
        -e HF_TOKEN_PATH=/existing-hf/token \
        --entrypoint python3 lmsysorg/sglang:latest -m sglang.launch_server \
        --model-path "$model" --host 127.0.0.1 --port 30001 --trust-remote-code \
        --context-length 8192 --mem-fraction-static 0.75 --max-running-requests 8 \
        >"output/$label/container-id.txt" 2>"output/$label/start-error.log"
    started=$?
    ready=0
    if [ "$started" -eq 0 ]; then
        for attempt in $(seq 1 180); do
            if curl -sf --max-time 3 http://127.0.0.1:30001/health >/dev/null; then ready=1; break; fi
            state=$(sudo docker inspect "$name" --format '{{.State.Running}}')
            if [ "$state" != true ]; then break; fi
            if [ $((attempt % 12)) -eq 0 ]; then echo "LOADING $model $(date -u +%FT%TZ)"; fi
            sleep 5
        done
    fi
    if [ "$ready" -eq 1 ]; then
        echo "TEST $model"
        $PY -u evals/model_gpu/run.py --model "$model" --output "output/$label" >"output/$label/client.log" 2>&1
        echo "TEST_EXIT $model $?"
    else
        echo "STARTUP_FAILED $model"
    fi
    sudo docker logs "$name" >"output/$label/server.log" 2>&1
    sudo docker inspect "$name" --format '{{json .Config.Cmd}}' >"output/$label/server-command.json"
    sudo docker stop "$name" >/dev/null 2>&1 || true
    # This directory was created solely for this run; preserve results separately.
    sudo rm -rf -- "$PWD/hf-cache/$label"
    echo "DONE $model $(date -u +%FT%TZ)"
done
