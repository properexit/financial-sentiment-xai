import torch
import torch.nn as nn
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from src.model import FinancialSentimentModel
from src.utils import (
    load_financial_phrasebank,
    tokenize_dataset,
    create_dataloader
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_training_epoch(model, dataloader, optimizer, criterion):
    """
    Run a single training epoch.

    This function is intentionally kept explicit (no Trainer API)
    so that gradient flow and optimization steps are transparent
    and easy to reason about.
    """
    model.train()
    cumulative_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        # Forward pass
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()

        cumulative_loss += loss.item()

    return cumulative_loss / len(dataloader)


def evaluate_on_validation_set(model, dataloader):
    """
    Evaluate the model on the validation split.

    We report accuracy here because Financial PhraseBank is
    a balanced, multi-class sentiment dataset where accuracy
    remains a meaningful metric.
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return accuracy_score(all_labels, all_predictions)


def main():
    print("Loading tokenizer and dataset...")

    # DistilBERT is chosen for its strong performance-to-size ratio.
    # It is sufficient for short financial statements and faster to fine-tune.
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train_ds, val_ds = load_financial_phrasebank()

    # Tokenization is done explicitly instead of using Trainer
    # to keep preprocessing fully visible and debuggable.
    train_ds = tokenize_dataset(train_ds, tokenizer)
    val_ds = tokenize_dataset(val_ds, tokenizer)

    train_loader = create_dataloader(
        train_ds, batch_size=16, shuffle=True
    )
    val_loader = create_dataloader(
        val_ds, batch_size=16, shuffle=False
    )

    print("Initializing model...")
    model = FinancialSentimentModel().to(DEVICE)

    # CrossEntropyLoss is appropriate because sentiment labels
    # are mutually exclusive (negative / neutral / positive).
    criterion = nn.CrossEntropyLoss()

    # AdamW is the standard optimizer for transformer fine-tuning.
    # A conservative learning rate helps avoid overfitting
    # on this relatively small dataset.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5
    )

    epochs = 3  # Sufficient for convergence without over-training

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss = run_training_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion
        )

        val_accuracy = evaluate_on_validation_set(
            model=model,
            dataloader=val_loader
        )

        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")

        # We intentionally do not apply early stopping here
        # to keep training behavior transparent and reproducible.

    torch.save(
        model.state_dict(),
        "data/processed/model.pt"
    )
    print("Model saved to data/processed/model.pt")


if __name__ == "__main__":
    main()