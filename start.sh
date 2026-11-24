#!/bin/bash
# Run the bot on a Mac/Linux computer. First time: installs everything.
cd "$(dirname "$0")"
[ -f .env ] || { cp .env.example .env; echo "Created .env - put your Alpaca PAPER keys in it, then run ./start.sh again."; exit 1; }
[ -d .venv ] || { python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; }
exec .venv/bin/python trade.py "$@"
