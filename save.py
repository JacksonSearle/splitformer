import os
import torch
import pickle


def save_model(save_name, model):

    # Make a models directory if there isn't one
    model_dir = "models"
    model_dir = os.path.join("..", model_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    # Make a folder in the models directory for the model if it doesn't exist
    model_dir = os.path.join(model_dir, save_name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # Save the model as a pytorch file
    torch.save(model.state_dict(), os.path.join(model_dir, f"{save_name}.pt"))

    # Save the model parameters as a pickle file
    with open(os.path.join(model_dir, f"{save_name}.pkl"), "wb") as f:
        model.tokenizer = None
        model.model = None
        pickle.dump(model, f)
 
    # Say where the model was saved
    print(f"\nModel saved to {model_dir} folder")
