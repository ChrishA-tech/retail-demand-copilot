#!/usr/bin/env bash
# UCI Online Retail II (Chen, 2012; CC BY 4.0). Mirror of the same 1,067,371 rows packaged by allanvc/onlineretail2.
set -e
mkdir -p data
[ -f data/onlineretail2.rda ] || curl -fsSL -o data/onlineretail2.rda \
  https://raw.githubusercontent.com/allanvc/onlineretail2/master/data/onlineretail2.rda
echo "data/onlineretail2.rda ready (you can also pass the UCI .xlsx with --data)"
