source .venv/bin/activate

python src/detection_model/train.py -m data.train_ratio=0.0001,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009
