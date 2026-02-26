source .venv/bin/activate

python src/detection_model/train.py -m data.train_ratio=1e-3 \
        data.augment_data=null
