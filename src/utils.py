import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

LABEL2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_financial_phrasebank():
    """
    Load the Financial PhraseBank dataset from a local CSV file.

    The dataset is intentionally stored locally to:
    - avoid HuggingFace script execution issues
    - make the project fully reproducible
    - keep data preprocessing explicit and auditable
    """

    df = pd.read_csv("data/raw/financial_phrasebank_allagree.csv")

    # Convert string sentiment labels to integer class IDs
    df["label"] = df["label"].map(LABEL2ID)

    # Convert to HuggingFace Dataset for efficient mapping/tokenization
    dataset = Dataset.from_pandas(df)

    # Simple random split; stratification is not critical at this scale
    split = dataset.train_test_split(
        test_size=0.2,
        seed=42
    )

    return split["train"], split["test"]


def tokenize_dataset(dataset, tokenizer, max_length=128):
    """
    Tokenize text samples using a pretrained tokenizer.

    Padding is fixed-length to simplify batching and explanation
    (token positions remain stable across samples).
    """

    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    tokenized = dataset.map(tokenize, batched=True)

    # Explicitly select tensor columns used during training
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    return tokenized


def create_dataloader(dataset, batch_size=16, shuffle=True):
    """
    Create a PyTorch DataLoader with an explicit collate function.

    A custom collate function avoids bugs caused by
    HuggingFace's default behavior when mixing tensors and metadata.
    """

    def collate_fn(batch):
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "label": torch.tensor(
                [x["label"] for x in batch],
                dtype=torch.long
            ),
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )