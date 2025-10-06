data-sythesis-research/
├─ configs/
│   └── add style params (use_style, style_vocab, style_dropout_p)
├─ src/
│   ├─ models/
│   │   └── model.py → insert style embedding + forward injection
│   ├─ train/
│   │   └── train.py → style-drop branch + modify loss call
│   ├─ sampler/
│   │   └── sample.py → implement conditional sampling + CFG
│   ├─ metrics/ 或 eval/
│   │   └── style_eval.py → style classifier evaluation
│   └─ data/
│       └── loader.py → read `style` field from dataset
├─ scripts/
│   ├─ train.sh → pass style-mode flags
│   └─ sample.sh → pass style and cfg parameters
├─ examples/
│   └── put a few sample prompts + style configs
├─ README.md → update usage with style control
└─ docs/ → Update docs.
