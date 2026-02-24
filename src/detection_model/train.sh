source .venv/bin/activate

python src/detection_model/train.py -m data.train_ratio=0.001
