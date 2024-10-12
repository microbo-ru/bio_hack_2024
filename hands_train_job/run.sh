#!/bin/sh
cd /workspace

wget https://storage.yandexcloud.net/biohack/main.py

python main.py --input "$1" --model "$2"

