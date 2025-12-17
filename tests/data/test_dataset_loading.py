import pytest

from data_process.data import get_entry_dataset


@pytest.mark.parametrize("split", ["train", "validation"])
def test_phish_email_dataset_loading(split):
    dataset = get_entry_dataset(
        name="phish-email",
        mode=split,
        cache_dir="data_phish/json",
        text_max_length=2048,
        num_proc=1,
    )

    assert dataset is not None
    assert len(dataset) > 0

    sample = dataset[0]
    assert "text_input_ids" in sample
    assert "style_caption_input_ids" in sample

    assert len(sample["text_input_ids"]) <= 2048
    assert len(sample["style_caption_input_ids"]) <= 2048
