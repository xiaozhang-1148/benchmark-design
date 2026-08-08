#!/usr/bin/env bash
# Backfill the 11 missing 第二批题目 into the Seed LaTeX CSV.
# Requires a valid Ark API key in ARK_API_KEY (UUID-style, not LTAI...).
set -euo pipefail
cd /home/baoquan/ocr-process/benchmark-design

if [[ -z "${ARK_API_KEY:-}" || "$ARK_API_KEY" == LTAI* ]]; then
  echo "请先: export ARK_API_KEY='你的方舟API_Key'（不要用 LTAI 开头的 AK）" >&2
  exit 2
fi

ONLY=(
  --only "20250825-0258-3866-a577-42ed1b311756_19"
  --only "20250827-0746-4910-63db-fb026a157065_19"
  --only "20250828-0742-3331-8c0e-60f42c401241_17"
  --only "20250829-0943-432b-04de-e316f5467146_16"
  --only "20250901-2343-4757-927b-89dd1c171615_17"
  --only "20250903-1336-5488-d68d-1fe9c9911732_15"
  --only "20250903-1336-5488-d68d-1fe9c9911732_19"
  --only "20250904-0316-3901-ffa0-f534d0588492_%e4%b8%89.17"
  --only "20250904-0321-09d8-4883-11af53938495_17"
  --only "20250915-1217-09f4-78c5-af7146901132_%e4%ba%8c.19"
  --only "20250921-2357-1664-8db6-2748b8427632_13"
)

.venv/bin/python tools/seed21pro_tools/src/qa_latex_transcribe/run_transcribe.py \
  "${ONLY[@]}" \
  --force-rerun \
  --workers 8

# Re-apply student rule: only fill when BOTH Q and A empty; clear otherwise.
.venv/bin/python tools/seed21pro_tools/fill_student_ocr.py

echo "done"
