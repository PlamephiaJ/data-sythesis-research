source .venv/bin/activate

python src/detection_model/train.py -m data.train_ratio=1e-3 \
        data.augment_data=data_phish/eval/origin/1e-3/augmentation_eval.jsonl
