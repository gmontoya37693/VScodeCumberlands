#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <operator> [asset_id]"
  exit 1
fi

operator="$1"
asset_id="${2:-}"
script_dir="$(cd -- "$(dirname -- "$0")/.." && pwd)"
python_bin="/opt/anaconda3/bin/python"
if [[ ! -x "$python_bin" ]]; then
  python_bin="python3"
fi

if [[ -n "$asset_id" ]]; then
  "$python_bin" "$script_dir/alc_item_sheet_tracker.py" schedule --operator "$operator" --asset-id "$asset_id"
else
  "$python_bin" "$script_dir/alc_item_sheet_tracker.py" schedule --operator "$operator"
fi
