from transformers import AutoTokenizer


class DataModule:
    def __init__(
        self,
        model_name="google/bert_uncased_L-2_H-128_A-2",
        batch_size=64,
        max_length=128,
    ):
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_data(self, example):
        return self.tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
