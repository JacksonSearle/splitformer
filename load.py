from train_model import TransformerModel, RetNetModel, SplitformerModel
import torch
import os
import pickle
from datasets import load_wikitext2


def load_model(model_name):
    
    # Make a path to the model_name folder
    model_dir = "models"
    model_dir = os.path.join("..", model_dir, model_name)

    # Load the model from the pickle file
    pytorch_model = torch.load(os.path.join(model_dir, f"{model_name}.pt"))
    
    # Load the model parameters from the pickle file
    with open(os.path.join(model_dir, f"{model_name}.pkl"), "rb") as f:
        model = pickle.load(f)

    # Grab the correct tokenizer
    _, _, _, tokenizer = load_wikitext2(model.max_seq_len, model.batch_size, model.tokens_per_pass)

    # Load the tokenizer
    model.tokenizer = tokenizer

    # Load the pytorch model
    model.model.load_state_dict(pytorch_model)