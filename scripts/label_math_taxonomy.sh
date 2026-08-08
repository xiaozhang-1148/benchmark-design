#!/usr/bin/env bash
# Label all CSV rows with L1/L2 taxonomy via Seed 2.1 Pro (one isolated thread per row).
set -euo pipefail
cd /home/baoquan/ocr-process/benchmark-design

if [[ -z "${ARK_API_KEY:-}" || "$ARK_API_KEY" == LTAI* ]]; then
  echo "请先: export ARK_API_KEY='你的方舟API_Key'（UUID 格式，不要用 LTAI 开头的 AK）" >&2
  exit 2
fi

# Optional rate-limit cap: ARK_LABEL_WORKERS=64 bash scripts/label_math_taxonomy.sh
.venv/bin/python tools/seed21pro_tools/label_math_taxonomy.py \
  --csv "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/tempt_data/第二批数据题目_seed21pro_latex.csv" \
  --taxonomy "/home/baoquan/ocr-process/benchmark-design/tools/seed21pro_tools/高中数学三级标签框架_一级与二级优化版_v0.2.md" \
  --resume \
  ${ARK_LABEL_WORKERS:+--max-workers "$ARK_LABEL_WORKERS"}

echo "done"
