#!/bin/bash

source .venv/bin/activate
# Scale by sigma is only used in absorb settings
python src/train/train.py model.scale_by_sigma=True graph.type=absorb
# python src/train/train.py model.scale_by_sigma=False graph.type=uniform
