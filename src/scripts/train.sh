#!/bin/bash

source .venv/bin/activate
python src/train/train.py model.scale_by_sigma=False
