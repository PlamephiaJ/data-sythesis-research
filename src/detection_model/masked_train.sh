#!/bin/bash

source .venv/bin/activate

python src/detection_model/masked_train.py -m data.mask_cluster_id=0 \
    data.masked_data_preserve_rate=0.001,0.005
