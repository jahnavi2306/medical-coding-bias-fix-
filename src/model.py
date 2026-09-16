import torch
import torch.nn as nn
from transformers import AutoModel


class PLM_ICD(nn.Module):
    """PubMedBERT encoder + label-wise attention + per-code classifier."""

    def __init__(self, model_name, num_labels, chunk_size=128, max_chunks=10):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.attention = nn.Linear(hidden_size, num_labels, bias=False)  # one query per code
        self.classifier = nn.Linear(hidden_size, num_labels)             # one output per code

        self.num_labels = num_labels
        self.max_chunks = max_chunks
        self.chunk_size = chunk_size

    def forward(self, input_ids):
        # input_ids: (batch_size, max_chunks, chunk_size)
        batch_size = input_ids.shape[0]

        flat_input = input_ids.view(-1, self.chunk_size)
        attention_mask = (flat_input != 0).long()

        outputs = self.encoder(input_ids=flat_input, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        hidden_size = token_embeddings.shape[-1]
        token_embeddings = token_embeddings.view(batch_size, self.max_chunks * self.chunk_size, hidden_size)

        # Label-wise attention: each code attends to the tokens relevant to it
        attention_scores = self.attention(token_embeddings)           # (batch, tokens, num_labels)
        attention_weights = torch.softmax(attention_scores, dim=1)
        label_representations = torch.einsum('btl,bth->blh', attention_weights, token_embeddings)

        logits = self.classifier.weight.mul(label_representations).sum(dim=2) + self.classifier.bias
        return logits
