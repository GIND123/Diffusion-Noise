#!/bin/bash
# Wait for all Zaratan jobs to drain, then collect, plot and push to Hugging Face.
cd "$(dirname "$0")"
ZR=../zaratan-run.sh
echo "[watch] waiting for jobs to finish ..."
while true; do
  n=$($ZR 'squeue -u $USER -h -o "%T" | wc -l' 2>/dev/null | tr -dc '0-9')
  [ -z "$n" ] && n=1
  echo "[watch] $n job(s) still in queue  ($(date +%H:%M:%S))"
  [ "$n" -eq 0 ] && break
  sleep 120
done
echo "[watch] all jobs done - collecting and pushing"
python3 push_to_hf.py
