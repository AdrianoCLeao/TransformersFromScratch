import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))  # Learnable scale parameter
        self.bias = nn.Parameter(torch.zeros(features))  # Learnable shift parameter

    def forward(self, x):
        # x shape: (batch, seq_len, hidden_size)
        mean = x.mean(dim=-1, keepdim=True)  # Calculate mean along the last dimension
        std = x.std(dim=-1, keepdim=True)  # Calculate standard deviation along the last dimension
        # Normalize the input using the calculated mean and std, and apply the learnable parameters
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # First linear transformation
        self.dropout = nn.Dropout(dropout)  # Dropout for regularization
        self.linear_2 = nn.Linear(d_ff, d_model)  # Second linear transformation

    def forward(self, x):
        # Apply first linear transformation, ReLU activation, dropout, and second linear transformation
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)  # Embedding layer with vocab size and embedding dimension

    def forward(self, x):
        # x shape: (batch, seq_len)
        # Embed the input and scale the embeddings by sqrt(d_model) as per the original Transformer paper
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # Dropout layer for regularization

        # Initialize positional encoding matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Create a position vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # Create a scaling factor vector of shape (d_model / 2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices

        # Add a batch dimension to the positional encoding matrix
        pe = pe.unsqueeze(0)  # Shape: (1, seq_len, d_model)

        # Register the positional encoding matrix as a buffer (non-learnable parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input tensor and apply dropout
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  # Shape: (batch, seq_len, d_model)
        return self.dropout(x)
    
class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # Dropout layer for regularization
        self.norm = LayerNormalization(features)  # Layer normalization for stabilizing the training

    def forward(self, x, sublayer):
        # Apply layer normalization, sublayer, dropout, and add the original input (residual connection)
        return x + self.dropout(sublayer(self.norm(x)))
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding vector size
        self.h = h  # Number of attention heads
        # Ensure d_model is divisible by the number of heads
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h  # Dimension of the vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Linear layer for query projection
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Linear layer for key projection
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Linear layer for value projection
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Linear layer for output projection
        self.dropout = nn.Dropout(dropout)  # Dropout layer for regularization

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Compute attention scores: (batch, h, seq_len, d_k) @ (batch, h, d_k, seq_len) -> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Apply mask (setting -inf where mask is 0) to avoid attending to certain positions
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        # Apply softmax to get attention weights: (batch, h, seq_len, seq_len)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # Compute the final attention output: (batch, h, seq_len, seq_len) @ (batch, h, seq_len, d_k) -> (batch, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # Project input queries, keys, and values using the corresponding linear layers
        query = self.w_q(q)  # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        key = self.w_k(k)    # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model) -> (batch, seq_len, d_model)

        # Reshape and transpose the input for multi-head attention: (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention using the static attention method
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Concatenate attention heads: (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Apply the output linear layer: (batch, seq_len, d_model) -> (batch, seq_len, d_model)  
        return self.w_o(x)
    
class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block  # Self-attention block
        self.feed_forward_block = feed_forward_block  # Feed-forward block
        # Residual connections with layer normalization
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # Apply the first residual connection around the self-attention block
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        # Apply the second residual connection around the feed-forward block
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers  # List of encoder blocks
        self.norm = LayerNormalization(features)  # Layer normalization applied after all encoder blocks

    def forward(self, x, mask):
        # Pass the input through each encoder block
        for layer in self.layers:
            x = layer(x, mask)
        # Apply layer normalization to the final output
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block  # Self-attention block for the decoder
        self.cross_attention_block = cross_attention_block  # Cross-attention block (attends to encoder output)
        self.feed_forward_block = feed_forward_block  # Feed-forward block
        # Residual connections with layer normalization for self-attention, cross-attention, and feed-forward blocks
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Apply the first residual connection around the self-attention block
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # Apply the second residual connection around the cross-attention block
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        # Apply the third residual connection around the feed-forward block
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers  # List of decoder blocks
        self.norm = LayerNormalization(features)  # Layer normalization applied after all decoder blocks

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Pass the input through each decoder block
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        # Apply layer normalization to the final output
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        # Linear layer to project from the model dimension to the vocabulary size
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> torch.Tensor:
        # Project the input tensor from (batch, seq_len, d_model) to (batch, seq_len, vocab_size)
        return self.proj(x)
    
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        # Initialize components of the Transformer model
        self.encoder = encoder  # Encoder part of the Transformer
        self.decoder = decoder  # Decoder part of the Transformer
        self.src_embed = src_embed  # Embedding layer for source input
        self.tgt_embed = tgt_embed  # Embedding layer for target input
        self.src_pos = src_pos  # Positional encoding for source input
        self.tgt_pos = tgt_pos  # Positional encoding for target input
        self.projection_layer = projection_layer  # Linear layer for projecting decoder output to vocabulary space

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode the source sequence into context representations.
        
        Parameters:
        - src: Source input tensor of shape (batch, seq_len)
        - src_mask: Mask tensor for the source input
        
        Returns:
        - Tensor of shape (batch, seq_len, d_model) representing encoded source sequence
        """
        # Embed and apply positional encoding to the source input
        src = self.src_embed(src)  # Convert source input to embedding space
        src = self.src_pos(src)    # Add positional encoding
        # Pass the encoded input through the encoder
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Decode the target sequence using encoder output and target masks.
        
        Parameters:
        - encoder_output: Encoded representation of source input
        - src_mask: Mask tensor for the source input
        - tgt: Target input tensor of shape (batch, seq_len)
        - tgt_mask: Mask tensor for the target input
        
        Returns:
        - Tensor of shape (batch, seq_len, d_model) representing decoded target sequence
        """
        # Embed and apply positional encoding to the target input
        tgt = self.tgt_embed(tgt)  # Convert target input to embedding space
        tgt = self.tgt_pos(tgt)    # Add positional encoding
        # Pass the target input through the decoder
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project the decoder output to vocabulary space.
        
        Parameters:
        - x: Tensor of shape (batch, seq_len, d_model) representing decoded target sequence
        
        Returns:
        - Tensor of shape (batch, seq_len, vocab_size) representing projected output
        """
        return self.projection_layer(x)  # Project to vocabulary size
    
def build_transformer(
    src_vocab_size: int, 
    tgt_vocab_size: int, 
    src_seq_len: int, 
    tgt_seq_len: int, 
    d_model: int = 512, 
    N: int = 6, 
    h: int = 8, 
    dropout: float = 0.1, 
    d_ff: int = 2048
) -> Transformer:
    """
    Build a Transformer model based on the specifications.

    Parameters:
    - src_vocab_size: Size of the source vocabulary
    - tgt_vocab_size: Size of the target vocabulary
    - src_seq_len: Length of the source sequences
    - tgt_seq_len: Length of the target sequences
    - d_model: Dimensionality of the model's hidden layers
    - N: Number of layers in the encoder and decoder
    - h: Number of attention heads
    - dropout: Dropout rate
    - d_ff: Dimensionality of the feed-forward network

    Returns:
    - Transformer model instance
    """
    
    # Create the embedding layers for source and target
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers for source and target
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create and initialize the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create and initialize the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder by stacking blocks
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer to map decoder outputs to vocabulary size
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the Transformer model with all components
    transformer = Transformer(
        encoder=encoder, 
        decoder=decoder, 
        src_embed=src_embed, 
        tgt_embed=tgt_embed, 
        src_pos=src_pos, 
        tgt_pos=tgt_pos, 
        projection_layer=projection_layer
    )
    
    # Initialize parameters using Xavier uniform initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer