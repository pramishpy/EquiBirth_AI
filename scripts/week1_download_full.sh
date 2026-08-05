#!/usr/bin/env bash
# Download the FULL per-year NBER natality CSVs (2018-2024), resumable, retrying, in parallel.
#
# Why curl -C - : the NBER server throttles/drops long single transfers. -C - resumes from the last
# byte already on disk, --retry re-attempts transient errors, --speed-time aborts a stalled socket so
# a retry can resume. We run one connection per year in parallel because the throttle is per-connection,
# so 7 parallel streams finish far faster than one serial stream.
#
# bash 3.2 compatible (macOS default): no associative arrays. Safe to re-run: complete files skip,
# partial files resume.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
mkdir -p "$RAW"

YEARS=(2018 2019 2020 2021 2022 2023 2024)
SIZES=(1873289723 1851708487 1776551100 1801003417 1804046181 2212345146 2232876130)

fetch() {
  local y="$1" want="$2"
  local url="https://data.nber.org/nvss/natality/csv/$y/natality${y}us.csv"
  local out="$RAW/natality${y}us.csv"
  local have; have=$( [ -f "$out" ] && stat -f%z "$out" 2>/dev/null || echo 0 )
  if [ "$have" -eq "$want" ]; then
    echo "$y: complete ($have bytes), skipping"
    return 0
  fi
  echo "$y: starting (have $have / want $want)"
  curl -sL --fail --retry 100 --retry-delay 5 --retry-all-errors \
       --speed-time 60 --speed-limit 1024 \
       -C - -o "$out" "$url"
  have=$( stat -f%z "$out" 2>/dev/null || echo 0 )
  if [ "$have" -eq "$want" ]; then
    echo "$y: OK ($have bytes)"
  else
    echo "$y: INCOMPLETE ($have / $want bytes) - re-run to resume"
  fi
}

pids=()
for i in "${!YEARS[@]}"; do
  fetch "${YEARS[$i]}" "${SIZES[$i]}" &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

echo "ALL DONE (fail=$fail)"
exit $fail
