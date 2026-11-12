#!/usr/bin/env bash
# Reproduce every result in results/ (except the LLM baseline, which costs money:
# run `geoloc llm` separately). Takes about an hour on a laptop CPU.
set -euo pipefail
cd "$(dirname "$0")/.."

geoloc download   # skip if data/raw already has the SQL dumps
geoloc prepare
geoloc users-social
geoloc users-text
geoloc tweets-text
geoloc users-embed
geoloc tweets-embed
geoloc report
