#!/usr/bin/env sh
# Start the stack on the NVIDIA GPU when Docker can use one, otherwise on CPU.
#
#   ./start.sh            start (or apply .env changes)
#   ./start.sh --build    rebuild images after code changes, then start
#
# The choice is saved as COMPOSE_FILE in .env, so later plain `docker compose ...`
# commands (exec, stop, logs, up) keep using the same GPU/CPU setup.
cd "$(dirname "$0")" || exit 1

GPU_FILES="docker-compose.yml:docker-compose.gpu.yml"
CPU_FILES="docker-compose.yml"

save_compose_file() {
    touch .env
    grep -v '^[[:space:]]*COMPOSE_FILE[[:space:]]*=' .env > .env.tmp
    echo "COMPOSE_FILE=$1" >> .env.tmp
    mv .env.tmp .env
}

use_gpu=no
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    use_gpu=yes
fi

if [ "$use_gpu" = yes ]; then
    echo "NVIDIA GPU found - starting with GPU support..."
    log=$(mktemp)
    { COMPOSE_FILE="$GPU_FILES" docker compose up -d "$@"; echo $? > "$log.rc"; } 2>&1 | tee "$log"
    rc=$(cat "$log.rc")
    if [ "$rc" -ne 0 ]; then
        # Fall back only when the failure is about the GPU; other failures are reported as they are
        if grep -qiE 'nvidia|gpu|device driver|could not select device' "$log"; then
            echo "Docker could not start with the GPU - falling back to CPU."
            use_gpu=no
        else
            echo "Startup failed (not a GPU problem) - see the output above."
            rm -f "$log" "$log.rc"
            exit "$rc"
        fi
    fi
    rm -f "$log" "$log.rc"
else
    echo "No NVIDIA GPU found - starting on CPU (answers will be slow)."
fi

if [ "$use_gpu" = yes ]; then
    save_compose_file "$GPU_FILES"
else
    COMPOSE_FILE="$CPU_FILES" docker compose up -d "$@" || exit $?
    save_compose_file "$CPU_FILES"
fi

echo
echo "Running on $( [ "$use_gpu" = yes ] && echo GPU || echo CPU ). App: http://localhost:3000"
echo "Check where the model runs:  docker compose exec ollama ollama ps"
