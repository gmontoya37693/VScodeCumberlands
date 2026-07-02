#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <operator> <as-of YYYY-MM-DD>"
  exit 1
fi

operator="$1"
as_of="$2"
script_dir="$(cd -- "$(dirname -- "$0")/.." && pwd)"
python_bin="/opt/anaconda3/bin/python"
if [[ ! -x "$python_bin" ]]; then
  python_bin="python3"
fi

"$python_bin" "$script_dir/alc_item_sheet_tracker.py" daily \
  --operator "$operator" \
  --as-of "$as_of"
