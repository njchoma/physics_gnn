#!/bin/bash

NYUDIR='/home/nc2201/research/GCNN/modelsNYU'
NERSCDIR='/home/nc2201/research/GCNN/modelsNERSC'
ICEDIR='/home/nc2201/research/GCNN/modelsICECUBE'

python3 script/summarize.py --path $NYUDIR
python3 script/summarize.py --path $NERSCDIR
python3 script/summarize.py --path $ICEDIR
