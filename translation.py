from pathlib import Path
from utils.utils import get_config, latest_weights_file_path 
from transformer.transformer import build_transformer
from tokenizers import Tokenizer
from datasets import load_dataset
from train.dataset import BilingualDataset
import torch
import sys

def translate(sentence: str):
    """
    Translates a given sentence from the source language to the target language using a pre-trained model.
    
    Args:
        sentence (str): Sentence to translate. Can also be an integer index to fetch from the dataset.
    
    Returns:
        str: Translated sentence in the target language.
    """
    # Define the device and load configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    config = get_config()
    
    # Load tokenizers for source and target languages
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))
    
    # Build the model and load pre-trained weights
    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config["seq_len"], config['seq_len'], d_model=config['d_model']).to(device)
    model_filename = latest_weights_file_path(config)
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])

    # Check if sentence is an index and retrieve corresponding text from dataset
    label = ""
    if sentence.isdigit():
        id = int(sentence)
        ds = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='all')
        ds = BilingualDataset(ds, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
        sentence = ds[id]['src_text']
        label = ds[id]["tgt_text"]
    
    seq_len = config['seq_len']

    # Prepare the source sentence for the model
    model.eval()
    with torch.no_grad():
        source = tokenizer_src.encode(sentence)
        source = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64), 
            torch.tensor(source.ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2), dtype=torch.int64)
        ], dim=0).to(device)
        source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
        encoder_output = model.encode(source, source_mask)

        # Initialize the decoder input with the SOS token
        decoder_input = torch.empty(1, 1).fill_(tokenizer_tgt.token_to_id('[SOS]')).type_as(source).to(device)

        # Print the source sentence and target start prompt
        if label:
            print(f"{f'ID: ':>12}{id}") 
        print(f"{f'SOURCE: ':>12}{sentence}")
        if label:
            print(f"{f'TARGET: ':>12}{label}") 
        print(f"{f'PREDICTED: ':>12}", end='')

        # Generate the translation word by word
        while decoder_input.size(1) < seq_len:
            # Build mask for the target sequence
            decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int).type_as(source_mask).to(device)
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

            # Project the output to the vocabulary space and get the next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

            # Print the translated word
            print(f"{tokenizer_tgt.decode([next_word.item()])}", end=' ')

            # Break if we predict the end-of-sequence token
            if next_word == tokenizer_tgt.token_to_id('[EOS]'):
                break

    # Convert token IDs to tokens and return the result
    return tokenizer_tgt.decode(decoder_input[0].tolist())
    
# Read sentence from command-line argument or use a default sentence
if __name__ == '__main__':
    sentence = sys.argv[1] if len(sys.argv) > 1 else "I like to study Natural Language Processing."
    print(translate(sentence))