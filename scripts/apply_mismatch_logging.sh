#!/usr/bin/env bash
# Apply method-A instrumentation: measure FSDP-vs-vLLM log_prob mismatch during real PPO training.
# Run on cluster: bash apply_mismatch_logging.sh
set -euo pipefail

VERL=~/TinyZero/verl
ROLLOUT="$VERL/workers/rollout/vllm_rollout/vllm_rollout.py"
WORKER="$VERL/workers/fsdp_workers.py"

echo "[1/4] restore originals from .bak (if present)"
for f in "$ROLLOUT" "$WORKER"; do
  if [[ -f "$f.bak" ]]; then
    cp "$f.bak" "$f"
    echo "  restored $f"
  else
    echo "  WARN: no backup for $f -- keeping current file"
  fi
done

echo "[2/4] snapshot current state as .pre_mismatch_log"
cp "$ROLLOUT" "$ROLLOUT.pre_mismatch_log"
cp "$WORKER"  "$WORKER.pre_mismatch_log"

echo "[3/4] patch $ROLLOUT -- add vllm_log_probs key (do NOT touch old_log_probs comment)"
python3 - <<'PY'
import re, pathlib, sys
p = pathlib.Path.home() / "TinyZero/verl/workers/rollout/vllm_rollout/vllm_rollout.py"
src = p.read_text()
marker = "# 'old_log_probs': log_probs,"
if marker not in src:
    sys.exit(f"ERROR: marker not found in {p}. Inspect file manually.")
if "'vllm_log_probs'" in src:
    print("  already patched, skipping")
else:
    new = src.replace(
        marker,
        marker + "\n                'vllm_log_probs': log_probs,  # KEEP vLLM log_probs for mismatch measurement",
        1,
    )
    p.write_text(new)
    print("  inserted vllm_log_probs line")
PY

echo "[4/4] patch $WORKER -- log FSDP-vs-vLLM diff in compute_log_prob"
python3 - <<'PY'
import pathlib, sys, re
p = pathlib.Path.home() / "TinyZero/verl/workers/fsdp_workers.py"
src = p.read_text()
if "mismatch/lp_abs_mean" in src:
    print("  already patched, skipping")
    sys.exit(0)

# Find the compute_log_prob method and inject measurement just before its return.
# Strategy: locate `def compute_log_prob` and the first `return` after it.
m = re.search(r"def compute_log_prob\(self,[^\n]*\):\n(?P<body>(?:[ \t].*\n)+?)(?P<ret>[ \t]+return [^\n]+\n)", src)
if not m:
    sys.exit("ERROR: could not locate compute_log_prob return. Patch manually.")

inject = (
    "        # --- BEGIN mismatch instrumentation ---\n"
    "        try:\n"
    "            if 'vllm_log_probs' in data.batch.keys() and 'old_log_probs' in data.batch.keys():\n"
    "                import torch\n"
    "                fsdp_lp = data.batch['old_log_probs']\n"
    "                vllm_lp = data.batch['vllm_log_probs']\n"
    "                if 'response_mask' in data.batch.keys():\n"
    "                    mask = data.batch['response_mask'].bool()\n"
    "                else:\n"
    "                    amask = data.batch['attention_mask']\n"
    "                    mask = amask[:, -fsdp_lp.shape[1]:].bool()\n"
    "                diff = (fsdp_lp - vllm_lp)[mask]\n"
    "                ratio = diff.exp()\n"
    "                mm = {\n"
    "                    'mismatch/lp_abs_mean': diff.abs().mean().item(),\n"
    "                    'mismatch/lp_abs_max':  diff.abs().max().item(),\n"
    "                    'mismatch/ratio_max':   ratio.max().item(),\n"
    "                    'mismatch/ratio_min':   ratio.min().item(),\n"
    "                    'mismatch/ratio_gt_1p2':(ratio > 1.2).float().mean().item(),\n"
    "                    'mismatch/ratio_lt_0p8':(ratio < 0.8).float().mean().item(),\n"
    "                    'mismatch/n_tokens':    float(mask.sum().item()),\n"
    "                }\n"
    "                if not hasattr(output, 'meta_info') or output.meta_info is None:\n"
    "                    output.meta_info = {}\n"
    "                output.meta_info.setdefault('metrics', {}).update(mm)\n"
    "        except Exception as _e:\n"
    "            print('[mismatch-log] skip:', _e)\n"
    "        # --- END mismatch instrumentation ---\n"
)

new = src[:m.start('ret')] + inject + src[m.start('ret'):]
p.write_text(new)
print("  inserted mismatch instrumentation before compute_log_prob return")
PY

echo "done. diff summary:"
echo "--- rollout ---"
diff "$ROLLOUT.pre_mismatch_log" "$ROLLOUT" || true
echo "--- worker ---"
diff "$WORKER.pre_mismatch_log" "$WORKER" | head -60 || true
