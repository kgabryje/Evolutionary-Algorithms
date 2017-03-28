#!/bin/bash

for f in *.svg; do 
inkscape -D -z --file=$f --export-pdf=${f::-4}.pdf --export-latex;
done
rm *.svg
