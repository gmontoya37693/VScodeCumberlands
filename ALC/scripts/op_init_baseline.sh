#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <operator> <go-live YYYY-MM-DD> <notes>"
  exit 1
fi

operator="$1"
go_live="$2"
notes="$3"
script_dir="$(cd -- "$(dirname -- "$0")/.." && pwd)"
python_bin="/opt/anaconda3/bin/python"
if [[ ! -x "$python_bin" ]]; then
  python_bin="python3"
fi

"$python_bin" "$script_dir/alc_item_sheet_tracker.py" init-baseline \
  --operator "$operator" \
  --go-live "$go_live" \
  --reset-posted-ledger \
  --reset-closed-periods \
  --notes "$notes"
