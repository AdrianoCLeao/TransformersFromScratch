from pathlib import Path

def get_config():
    """
    Returns a dictionary containing configuration settings for the model training.

    Returns:
        dict: A dictionary with configuration parameters such as batch size, number of epochs,
              learning rate, sequence length, model dimensions, datasource, languages, and paths.
    """
    return {
        "batch_size": 16,  # Number of samples per batch
        "num_epochs": 1,   # Number of training epochs
        "lr": 10**-4,      # Learning rate for optimization
        "seq_len": 250,    # Sequence length for input data
        "d_model": 256,    # Dimensionality of the model's hidden layers
        "datasource": 'opus_books',  # Name of the datasource for the training data
        "lang_src": "en", # Source language code
        "lang_tgt": "pt", # Target language code
        "model_folder": "weights",   # Folder where model weights are stored
        "model_basename": "tmodel_", # Base name for model weight files
        "preload": "latest",         # Preload setting for model weights
        "tokenizer_file": "tokenizer_{0}.json", # Filename pattern for tokenizer
        "experiment_name": "runs/tmodel" # Directory for saving experiment results
    }

def get_weights_file_path(config, epoch: str):
    """
    Constructs the file path for the model weights file for a specific epoch.

    Args:
        config (dict): Configuration settings containing paths and filenames.
        epoch (str): Epoch number to include in the filename.

    Returns:
        str: The full path to the model weights file for the specified epoch.
    """
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
    """
    Finds the most recent weights file in the weights folder.

    Args:
        config (dict): Configuration settings containing paths and filenames.

    Returns:
        str: The full path to the most recent weights file, or None if no weights files are found.
    """
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))  # List all matching files
    if len(weights_files) == 0:
        return None  # No weights files found
    weights_files.sort()  # Sort files by name (assuming they include timestamps)
    return str(weights_files[-1])  # Return the most recent weights file
