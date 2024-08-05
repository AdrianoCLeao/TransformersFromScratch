# Transformer Model Documentation

## Overview

This repository contains an implementation of a Transformer model in PyTorch. The Transformer model consists of several key components: Layer Normalization, Multi-Head Attention, Feed-Forward Networks, and Encoder-Decoder architecture. Each component and function within the model is detailed below.

## Components

### LayerNormalization

- **Class:** `LayerNormalization`
- **Description:** Applies layer normalization to the input tensor.
- **Constructor:**
  - `features (int)`: Number of features in the input tensor.
  - `eps (float)`: A small constant to avoid division by zero.
- **Methods:**
  - `forward(x)`: Normalizes the input tensor `x`.

### FeedForwardBlock

- **Class:** `FeedForwardBlock`
- **Description:** Applies two linear transformations with ReLU activation in between, followed by dropout for regularization.
- **Constructor:**
  - `d_model (int)`: Dimensionality of the model's hidden layers.
  - `d_ff (int)`: Dimensionality of the feed-forward network.
  - `dropout (float)`: Dropout rate for regularization.
- **Methods:**
  - `forward(x)`: Applies the feed-forward network to the input tensor `x`.

### InputEmbeddings

- **Class:** `InputEmbeddings`
- **Description:** Converts input tokens into dense embeddings.
- **Constructor:**
  - `d_model (int)`: Dimensionality of the embeddings.
  - `vocab_size (int)`: Size of the vocabulary.
- **Methods:**
  - `forward(x)`: Converts input token indices `x` into embeddings and scales them by the square root of `d_model`.

### PositionalEncoding

- **Class:** `PositionalEncoding`
- **Description:** Adds positional encodings to input embeddings to provide information about the position of tokens in the sequence.
- **Constructor:**
  - `d_model (int)`: Dimensionality of the positional encodings.
  - `seq_len (int)`: Length of the input sequences.
  - `dropout (float)`: Dropout rate for regularization.
- **Methods:**
  - `forward(x)`: Adds positional encodings to the input tensor `x` and applies dropout.

### ResidualConnection

- **Class:** `ResidualConnection`
- **Description:** Applies residual connections with layer normalization around a given sublayer.
- **Constructor:**
  - `features (int)`: Number of features in the input tensor.
  - `dropout (float)`: Dropout rate for regularization.
- **Methods:**
  - `forward(x, sublayer)`: Applies residual connection and normalization to the input tensor `x`.

### MultiHeadAttentionBlock

- **Class:** `MultiHeadAttentionBlock`
- **Description:** Applies multi-head self-attention to the input tensor.
- **Constructor:**
  - `d_model (int)`: Dimensionality of the model's hidden layers.
  - `h (int)`: Number of attention heads.
  - `dropout (float)`: Dropout rate for regularization.
- **Methods:**
  - `attention(query, key, value, mask, dropout)`: Computes the attention scores and output.
  - `forward(q, k, v, mask)`: Applies multi-head attention to the input tensors `q`, `k`, and `v`.

### EncoderBlock

- **Class:** `EncoderBlock`
- **Description:** Applies a single encoder block, consisting of self-attention and feed-forward sub-layers with residual connections.
- **Constructor:**
  - `features (int)`: Number of features in the input tensor.
  - `self_attention_block (MultiHeadAttentionBlock)`: Self-attention block.
  - `feed_forward_block (FeedForwardBlock)`: Feed-forward block.
  - `dropout (float)`: Dropout rate for regularization.
- **Methods:**
  - `forward(x, src_mask)`: Applies self-attention and feed-forward blocks with residual connections.

### Encoder

- **Class:** `Encoder`
- **Description:** Stacks multiple encoder blocks and applies layer normalization to the final output.
- **Constructor:**
  - `features (int)`: Number of features in the input tensor.
  - `layers (nn.ModuleList)`: List of encoder blocks.
- **Methods:**
  - `forward(x, mask)`: Passes the input through each encoder block and applies final layer normalization.

### DecoderBlock

- **Class:** `DecoderBlock`
- **Description:** Applies a single decoder block, consisting of self-attention, cross-attention, and feed-forward sub-layers with residual connections.
- **Constructor:**
  - `features (int)`: Number of features in the input tensor.
  - `self_attention_block (MultiHeadAttentionBlock)`: Self-attention block for the decoder.
  - `cross_attention_block (MultiHeadAttentionBlock)`: Cross-attention block.
  - `feed_forward_block (FeedForwardBlock)`: Feed-forward block.
  - `dropout (float)`: Dropout rate for regularization.
- **Methods:**
  - `forward(x, encoder_output, src_mask, tgt_mask)`: Applies self-attention, cross-attention, and feed-forward blocks with residual connections.

### Decoder

- **Class:** `Decoder`
- **Description:** Stacks multiple decoder blocks and applies layer normalization to the final output.
- **Constructor:**
  - `features (int)`: Number of features in the input tensor.
  - `layers (nn.ModuleList)`: List of decoder blocks.
- **Methods:**
  - `forward(x, encoder_output, src_mask, tgt_mask)`: Passes the input through each decoder block and applies final layer normalization.

### ProjectionLayer

- **Class:** `ProjectionLayer`
- **Description:** Projects the decoder output to the vocabulary size.
- **Constructor:**
  - `d_model (int)`: Dimensionality of the model's hidden layers.
  - `vocab_size (int)`: Size of the target vocabulary.
- **Methods:**
  - `forward(x)`: Projects the input tensor `x` to the vocabulary space.

### Transformer

- **Class:** `Transformer`
- **Description:** Combines encoder, decoder, and other components to form the full Transformer model.
- **Constructor:**
  - `encoder (Encoder)`: Encoder part of the Transformer.
  - `decoder (Decoder)`: Decoder part of the Transformer.
  - `src_embed (InputEmbeddings)`: Embedding layer for source input.
  - `tgt_embed (InputEmbeddings)`: Embedding layer for target input.
  - `src_pos (PositionalEncoding)`: Positional encoding for source input.
  - `tgt_pos (PositionalEncoding)`: Positional encoding for target input.
  - `projection_layer (ProjectionLayer)`: Linear layer for projecting decoder output to vocabulary space.
- **Methods:**
  - `encode(src, src_mask)`: Encodes the source sequence into context representations.
  - `decode(encoder_output, src_mask, tgt, tgt_mask)`: Decodes the target sequence using encoder output and target masks.
  - `project(x)`: Projects the decoder output to vocabulary space.

## Build Function

### build_transformer

- **Function:** `build_transformer`
- **Description:** Constructs a Transformer model based on the specified parameters.
- **Parameters:**
  - `src_vocab_size (int)`: Size of the source vocabulary.
  - `tgt_vocab_size (int)`: Size of the target vocabulary.
  - `src_seq_len (int)`: Length of the source sequences.
  - `tgt_seq_len (int)`: Length of the target sequences.
  - `d_model (int)`: Dimensionality of the model's hidden layers.
  - `N (int)`: Number of layers in the encoder and decoder.
  - `h (int)`: Number of attention heads.
  - `dropout (float)`: Dropout rate.
  - `d_ff (int)`: Dimensionality of the feed-forward network.
- **Returns:** A Transformer model instance.

