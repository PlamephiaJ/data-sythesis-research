source .venv/bin/activate

python src/detection_model/train.py -m data.train_ratio=0.000001,0.00001,0.0001,0.001,0.01,0.1
