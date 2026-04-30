#!/usr/bin/env bash
# Istrina visus 3Lab eksperimentu rezultatus (summary.csv, kreives, confusion matrix,
# pavyzdziu prognozes ir background-run zymes).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="$DIR/rezultatai"

echo "Valoma: $RES/images/ ir $RES/timeseries/"
rm -rf "$RES/images" "$RES/timeseries"
mkdir -p "$RES/images" "$RES/timeseries"

echo "Salinami /tmp zymejimo failai ir logai"
rm -f /tmp/main_images_done /tmp/main_ts_done \
      /tmp/main_images.log /tmp/main_ts.log \
      /tmp/main_images_resume.log /tmp/setsid_launch.log

echo "Atlikta."
