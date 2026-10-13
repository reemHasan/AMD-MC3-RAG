# Source this in EVERY new terminal, from the project root:   . ./env.sh
# (exported variables do not survive a new terminal or a lab restart)
export MC3_MODEL_DIR="$PWD/models/vlm"
export MC3_INDEX_DIR="$PWD/work/index"
export MC2_OUTPUT_DIR="$PWD/work/output"
export MC3_SOCK=/tmp/mc3_rag.sock
export HF_HUB_OFFLINE=1
mkdir -p "$PWD/work"
echo "MC3_MODEL_DIR=$MC3_MODEL_DIR"
[ -f "$MC3_MODEL_DIR/config.json" ] && echo "model folder OK" || echo "WARNING: no config.json in $MC3_MODEL_DIR - download the weights first"
