import torch
from typing import Optional

class EmbeddingLayerWithDropout(torch.nn.Module):
  """
    Dropout-regularized embedding layer for discrete input sequences.

    This layer applies dropout directly to the word embedding matrix rather than the output vectors,
    effectively dropping entire word types (not tokens) during training. It is inspired by dropout
    techniques for continuous inputs, where applying noise to the input layer helps prevent overfitting.

    In NLP models, the embedding matrix is often the largest and most overfit-prone layer, yet rarely
    regularized. This layer addresses that by stochastically zeroing out rows in the embedding matrix
    — corresponding to word types — and applying the same dropout mask across the entire input
    sequence. As a result, if a word is dropped once, it is dropped for all its occurrences in the
    sequence.

    This can be interpreted as:
        - A form of word-type dropout (e.g., dropping "the" from an entire sentence).
        - A method that encourages the model to rely less on individual words and more on context.
        - An implicit distribution placed over the embedding matrix, approximating integration over it.

    Parameters:
        num_embeddings (int): Size of the vocabulary (number of unique tokens).
        embedding_dim (int): Dimensionality of the embeddings.
        dropout (float, optional): Dropout probability (default = 0.1). During training, word types
            are randomly dropped with this probability.
        padding_idx (int, optional): Index of the padding token in the vocabulary, if any.
        max_norm (float, optional): If given, renormalize embeddings to have a norm less than or
            equal to this value.
        norm_type (float, optional): Type of norm to use when renormalizing (default = 2.0).
        scale_grad_by_freq (bool, optional): Scale gradients by the frequency of words in the batch.
        sparse (bool, optional): If True, gradient w.r.t. weight matrix will be sparse.

    Note:
        - Dropout is applied at the word **type** level, not per occurrence (token), so the same word
          will be dropped consistently across the input.
        - During evaluation (`model.eval()`), dropout is disabled and all embeddings are used.
    """
  def __init__(
    self,
    num_embeddings: int,
    embedding_dim: int,
    dropout: float = 0.1,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False
  ):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.embedding = torch.nn.Embedding(
      num_embeddings, embedding_dim,
      padding_idx=padding_idx,
      max_norm=max_norm,
      norm_type=norm_type,
      scale_grad_by_freq=scale_grad_by_freq,
      sparse=sparse
    )
    self.dropout = dropout

  def forward(
    self,
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None
  ) -> torch.Tensor:
    """
      Inputs:
          input (LongTensor): Tensor of word indices with shape (batch_size, sequence_length).
          scale (Tensor, optional): Optional scaling tensor broadcastable to the embedding weights.

      Returns:
          Tensor of shape (batch_size, sequence_length, embedding_dim) containing the embedded vectors
          after applying dropout to the embedding weights.
    """
    if self.training and self.dropout > 0:
      mask = self.embedding.weight.new_empty((self.num_embeddings, self.embedding_dim))
      mask.bernoulli_(1 - self.dropout)
      mask = mask.div_(1 - self.dropout)
      masked_weight = self.embedding.weight * mask
    else:
      masked_weight = self.embedding.weight

    if scale is not None:
      masked_weight = scale.expand_as(masked_weight) * masked_weight

    return torch.nn.functional.embedding(
      input, masked_weight,
      self.embedding.padding_idx, self.embedding.max_norm,
      self.embedding.norm_type, self.embedding.scale_grad_by_freq,
      self.embedding.sparse
    )