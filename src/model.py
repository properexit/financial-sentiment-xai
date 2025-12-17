import torch
import torch.nn as nn
from transformers import DistilBertModel


class FinancialSentimentModel(nn.Module):
    """
    Lightweight transformer based classifier for financial sentiment analysis.

    Design choices:
    - DistilBERT is used instead of full BERT to reduce compute cost
      while retaining strong language understanding.
    - A simple linear classification head is sufficient for this task
      and makes gradient-based explanations easier to interpret.
    """

    def __init__(self, num_labels: int = 3):
        super().__init__()

        # Pretrained encoder provides contextual token representations
        self.encoder = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )

        hidden_size = self.encoder.config.hidden_size

        # Classification head maps sentence representation -> sentiment class
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through encoder and classification head.

        Args:
            input_ids: Tensor of token IDs (batch_size, seq_len)
            attention_mask: Tensor indicating valid tokens

        Returns:
            logits: Unnormalized class scores (batch_size, num_labels)
        """

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # DistilBERT does not provide a pooled output.
        # We use the first token ([CLS]-equivalent) as a
        # sentence-level representation, consistent with training.
        sentence_embedding = outputs.last_hidden_state[:, 0]

        logits = self.classifier(sentence_embedding)

        return logits