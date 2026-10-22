#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

if ss -ltn | grep -q ':5000 '; then
    echo "Inventory server already running. Open: http://127.0.0.1:5000"
    exit 0
fi

exec python src/web_bag_inventory.py --host 127.0.0.1 --port 5000
