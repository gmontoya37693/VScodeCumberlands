#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <operator> <month YYYY-MM> [billing_day]"
  exit 1
fi

operator="$1"
month="$2"
billing_day="${3:-22}"
script_dir="$(cd -- "$(dirname -- "$0")/.." && pwd)"
python_bin="/opt/anaconda3/bin/python"
if [[ ! -x "$python_bin" ]]; then
  python_bin="python3"
fi

echo ""
echo "______________"
echo ""

"$python_bin" "$script_dir/alc_item_sheet_tracker.py" invoice \
  --operator "$operator" \
  --month "$month" \
  --billing-day "$billing_day" \
  --output "$script_dir/invoices_${month}.csv"

echo ""
echo "______________"
echo ""
