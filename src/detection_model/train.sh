source .venv/bin/activate

python src/detection_model/train.py -m data.train_ratio=0.1 \
        data.augment_data=data_phish/eval/0.001/augmented_0.75.jsonl,data_phish/eval/0.001/augmented_0.5.jsonl,data_phish/eval/0.001/augmented_0.25.jsonl,data_phish/eval/0.001/augmented_0.1.jsonl,data_phish/eval/0.001/augmented_0.01.jsonl
