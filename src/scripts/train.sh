#!/bin/bash

source .venv/bin/activate
python src/train/train.py graph.type=uniform model=small model.scale_by_sigma=False
