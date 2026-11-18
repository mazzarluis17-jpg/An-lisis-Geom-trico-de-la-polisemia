#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p build
name=articulo_polisemia_definitivo
tex() {
    xelatex -interaction=nonstopmode -halt-on-error -file-line-error -jobname="$name" -output-directory=build main.tex
}
tex > build/xelatex_1.txt 2>&1
biber --input-directory=build --output-directory=build "$name" > build/biber.txt 2>&1
tex > build/xelatex_2.txt 2>&1
tex > build/xelatex_3.txt 2>&1
