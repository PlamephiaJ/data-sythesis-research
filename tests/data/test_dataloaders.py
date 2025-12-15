from omegaconf import OmegaConf

from data_process.data import get_dataloaders


cfg = OmegaConf.create(
    {
        "ngpus": 2,
        "training": {
            "batch_size": 8,
            "accum": 1,
        },
        "eval": {
            "batch_size": 8,
        },
        "data": {
            "format": "entry",
            "num_proc": 4,
            "max_length": 2048,
            "trainset": {
                "name": "phish-email",
                "cache_dir": "data_phish/json",
            },
            "validset": {
                "name": "phish-email",
                "cache_dir": "data_phish/json",
            },
        },
        "model": {
            "length": 1024,
        },
    }
)

print("=== CONFIG ===")
print(OmegaConf.to_yaml(cfg))

get_dataloaders(cfg, distributed=False)

print("=== SHOULD NOT REACH HERE ===")
