#!/usr/bin/env bash
# Paleidzia abi uzduotis (main.py ir main_ts.py) lygiagreciai, atskirti nuo terminalo.
# Naudojimas:
#   ./run_all.sh                  # numatytieji epochs (image=20, ts=50)
#   ./run_all.sh 25 60            # image_epochs=25, ts_epochs=60
#
# Po paleidimo procesai veikia foniniu rezimu (setsid + disown), todel terminalo
# uzdarymas ju nestabdo. Progresui sekti naudokite:
#   tail -f /tmp/main_images.log
#   tail -f /tmp/main_ts.log
#   ls /tmp/main_images_done /tmp/main_ts_done   # abudu egzistavimas = uzbaigta
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_ACTIVATE="$DIR/../env/bin/activate"

IMG_EPOCHS="${1:-20}"
TS_EPOCHS="${2:-50}"

if [[ ! -f "$ENV_ACTIVATE" ]]; then
    echo "ERR: nerasta venv aktyvacija: $ENV_ACTIVATE" >&2
    exit 1
fi

echo "Paleidziamos abi uzduotys lygiagreciai (image=$IMG_EPOCHS ep., ts=$TS_EPOCHS ep.)"
echo "Logai:"
echo "  /tmp/main_images.log"
echo "  /tmp/main_ts.log"

# Pasalinam senas baigties zymes, kad ju buvimas reikstu naujo paleidimo pabaiga.
rm -f /tmp/main_images_done /tmp/main_ts_done

setsid bash -c "
    cd '$DIR'
    source '$ENV_ACTIVATE'
    python main.py    '$IMG_EPOCHS' fresh > /tmp/main_images.log 2>&1 && touch /tmp/main_images_done &
    python main_ts.py '$TS_EPOCHS'  fresh > /tmp/main_ts.log     2>&1 && touch /tmp/main_ts_done     &
    wait
" < /dev/null > /dev/null 2>&1 & disown

sleep 1
echo "PIDs:"
pgrep -af "python main" || echo "(nepavyko rasti procesu - patikrinkite logus)"
echo "Atskyrimas atliktas. Laukti pabaigos: 'ls /tmp/main_*_done' kol abudu atsiras."
