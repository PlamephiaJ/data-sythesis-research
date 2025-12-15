from data_process.data import get_entry_dataset


train_set = get_entry_dataset(
    "phish-email",
    "train",
    cache_dir="data_phish/json",
    max_length=2048,
    num_proc=120,
)

# valid_set = get_entry_dataset(
#             "phish-email",
#             "validation",
#             cache_dir="data_phish/json",
#             max_length=2048,
#             num_proc=120,
#         )

print(f"Train set size: {len(train_set)}")
# print(f"Validation set size: {len(valid_set)}")
