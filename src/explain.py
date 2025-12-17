import torch
import numpy as np
from transformers import AutoTokenizer

from src.model import FinancialSentimentModel
from src.utils import ID2LABEL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GradientInputExplainer:
    """
    Computes token-level attributions using Gradient × Input.

    This explainer is intentionally simple and model-specific:
    - no external XAI libraries
    - no attention-based heuristics
    - gradients are taken directly w.r.t. input embeddings

    The goal is faithfulness and transparency rather than convenience.
    """

    def __init__(self, model_name="distilbert-base-uncased"):
        # Tokenizer must match the model used during training
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load trained model weights
        self.model = FinancialSentimentModel()
        self.model.load_state_dict(
            torch.load("data/processed/model.pt", map_location=DEVICE)
        )

        self.model.to(DEVICE)
        self.model.eval()  

    def explain(self, text: str):
        """
        Generate a Gradient × Input explanation for a single sentence.

        Returns:
        - tokens: list of wordpiece tokens
        - scores: normalized importance scores per token
        - prediction: predicted sentiment label (string)
        """

        # Tokenize input text
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128  # sufficient for short financial statements
        )

        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        # We extract embeddings explicitly so that we can attach gradients.
        # Embeddings are non-leaf tensors, so retain_grad() is required.
        embeddings = self.model.encoder.embeddings.word_embeddings(input_ids)
        embeddings.retain_grad()
        embeddings.requires_grad_(True)

        # Forward pass through the encoder using embeddings directly
        encoder_outputs = self.model.encoder(
            inputs_embeds=embeddings,
            attention_mask=attention_mask
        )

        # We use the [CLS] representation for sentence-level classification,
        # consistent with how the model was trained.
        pooled_representation = encoder_outputs.last_hidden_state[:, 0, :]
        logits = self.model.classifier(pooled_representation)

        predicted_class = torch.argmax(logits, dim=1).item()

        # Backpropagate only the predicted class score
        # This yields class-specific attributions.
        self.model.zero_grad()
        logits[0, predicted_class].backward()

        gradients = embeddings.grad

        # Gradient × Input attribution:
        # element-wise product, then L2 norm across embedding dimensions
        token_importance = torch.norm(
            gradients * embeddings, dim=-1
        ).squeeze(0)

        tokens = self.tokenizer.convert_ids_to_tokens(
            input_ids.squeeze(0)
        )

        # Normalize scores for easier visualization
        scores = token_importance.detach().cpu().numpy()
        scores = scores / (scores.max() + 1e-8)

        return {
            "tokens": tokens,
            "scores": scores.tolist(),
            "prediction": ID2LABEL[predicted_class]
        }