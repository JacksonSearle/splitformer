import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataset
 
 
from argparse import ArgumentParser
from torchscale.architecture.config import DecoderConfig, RetNetConfig
from torchscale.architecture.decoder import Decoder
from torchscale.architecture.retnet import RetNetDecoder 
from torchscale.architecture.splitformer import SplitformerDecoder

from save import save_model


import os

import time


import pickle


from torchinfo import summary as model_summary
 
 
from datasets import load_wikitext2
 
 
from tqdm import tqdm
 
 
from tabulate import tabulate
 
 
class RetNetModel(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            value_embed_dim: int,
            retention_heads: int,
            ffn_dim: int,
            layers: int,
            dropout: float,
            activation_dropout: float,
            vocab_size: int,
            checkpoint_activations: bool,
            fsdp: bool,
            max_seq_len: int):
        """ Use parameters to create corresponding RetNet model
        Args:
            embed_dim (int): Dimension size of each embedded token.
            value_embed_dim (int): Value embed dimension size.
            retention_heads (int): Number of retention heads in MSR module.
            ffn_dim (int): Hidden layer size of Feed Forward Network (FFN).
            layers (int): Number of retention network layers.
            dropout (float): Probability of an element to be zeroed during dropout.
            activation_dropout (float): Probability of an element to be zeroed
                during dropout after activation between FFN layers.
            vocab_size (int): Maximum vocabulary size (number of unique tokens in
                vocabulary.
            checkpoint_activations (bool): Whether to perform checkpointing or not
                (done with the FairScale library).
            fsdp (bool): Whether to shard Module parameters across data parallel
                workers or not (with the FairScale library).
            max_seq_len (int): Size of context window.
        """
        super().__init__()
 
 
        self.model_params = {
                "embed_dim": embed_dim,
                "value_embed_dim": value_embed_dim,
                "retention_heads": retention_heads,
                "ffn_dim": ffn_dim,
                "layers": layers,
                "dropout": dropout,
                "activation_dropout": activation_dropout,
                "vocab_size": vocab_size,
                "checkpoint_activations": checkpoint_activations,
                "fsdp": fsdp,
                "max_seq_len": max_seq_len
                }
 
 
        config = RetNetConfig(
                decoder_embed_dim=embed_dim,
                decoder_value_embed_dim=value_embed_dim,
                decoder_retention_heads=retention_heads,
                decoder_ffn_embed_dim=ffn_dim,
                decoder_layers=layers,
                dropout=dropout,
                activation_dropout=activation_dropout,
                vocab_size=vocab_size,
                checkpoint_activations=checkpoint_activations,
                fsdp=fsdp)
 
        # Save max_seq_len for padding later
        self.max_seq_len = max_seq_len
 
 
        # Save vocab_size for final dimensions later
        self.vocab_size = vocab_size
 
 
        # Create embeddings with index 0 representing padding
        self.text_embeddings = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embed_dim,
                padding_idx=1)
 
 
        #TODO: Check that we are masking correctly
        self.model = RetNetDecoder(config, embed_tokens=self.text_embeddings)
 
 
    def forward(self, x: torch.Tensor, encoder_padding_mask=False) -> torch.Tensor:
        logits, other_stuff = self.model(x, encoder_padding_mask=encoder_padding_mask)
        return logits
 
 
    def generate_text(self, start_string, generation_length=100, device='cuda'):
        # Evaluation mode
        self.model.eval()
        self.model.to(device)
 
 
        # Convert start string to numbers
        input_eval = self.tokenizer.stoi(start_string)
        print(input_eval)
        input_eval = torch.tensor(input_eval).unsqueeze(0).to(device)
 
 
        # Empty list to store generated text
        text_generated = []
 
 
        # No gradients needed
        with torch.no_grad():
            for _ in range(generation_length):
                predictions = self.forward(input_eval)

                predictions = predictions[:, -1]
                # Apply softmax to get probabilities
                predictions = F.softmax(predictions, dim=-1)

                # Get the top-k predicted words
                top_k = 5
                topk_values, topk_indices = torch.topk(predictions, k=top_k, dim=-1)
                # Sample from topk_values
                predicted_id = torch.multinomial(topk_values.squeeze(), num_samples=1)
                # Get the corresponding topk_indices
                predicted_id = torch.gather(topk_indices.squeeze(), dim=-1, index=predicted_id)

                # Add predicted word to the input (to be used as next input sequence)
                input_eval = torch.cat([input_eval, predicted_id.unsqueeze(-1)], dim=-1)

                # Convert predicted word id to word
                predicted_word = self.tokenizer.itos(predicted_id.tolist())

                text_generated.append(predicted_word)
 
 
        return start_string + ' ' + ' '.join(text_generated)
 

class TransformerModel(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            value_embed_dim: int,
            attention_heads: int,
            ffn_dim: int,
            layers: int,
            dropout: float,
            activation_dropout: float,
            vocab_size: int,
            checkpoint_activations: bool,
            fsdp: bool,
            max_seq_len: int):
        """ Use parameters to create corresponding RetNet model
        Args:
            embed_dim (int): Dimension size of each embedded token.
            value_embed_dim (int): Value embed dimension size.
            attention_heads (int): Number of attention heads in MHA module.
            ffn_dim (int): Hidden layer size of Feed Forward Network (FFN).
            layers (int): Number of retention network layers.
            dropout (float): Probability of an element to be zeroed during dropout.
            activation_dropout (float): Probability of an element to be zeroed
                during dropout after activation between FFN layers.
            vocab_size (int): Maximum vocabulary size (number of unique tokens in
                vocabulary.
            checkpoint_activations (bool): Whether to perform checkpointing or not
                (done with the FairScale library).
            fsdp (bool): Whether to shard Module parameters across data parallel
                workers or not (with the FairScale library).
            max_seq_len (int): Size of context window.
        """
        super().__init__()
 
 
        self.model_params = {
                "embed_dim": embed_dim,
                "value_embed_dim": value_embed_dim,
                "attention_heads": attention_heads,
                "ffn_dim": ffn_dim,
                "layers": layers,
                "dropout": dropout,
                "activation_dropout": activation_dropout,
                "vocab_size": vocab_size,
                "checkpoint_activations": checkpoint_activations,
                "fsdp": fsdp,
                "max_seq_len": max_seq_len,
                }
 
 
        config = DecoderConfig(
                decoder_embed_dim=embed_dim,
                decoder_value_embed_dim=value_embed_dim,
                decoder_attention_heads=attention_heads,
                decoder_ffn_embed_dim=ffn_dim,
                decoder_layers=layers,
                dropout=dropout,
                activation_dropout=activation_dropout,
                vocab_size=vocab_size,
                checkpoint_activations=checkpoint_activations,
                fsdp=fsdp)
 
 
        # Save max_seq_len for padding later
        self.max_seq_len = max_seq_len
 
 
        # Save vocab_size for final dimensions later
        self.vocab_size = vocab_size
 
 
        # Create embeddings with index 0 representing padding
        self.text_embeddings = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embed_dim,
                padding_idx=1)
 
 
        self.model = Decoder(config, embed_tokens=self.text_embeddings)
 
 
    def forward(self, x: torch.Tensor, encoder_padding_mask=False) -> torch.Tensor:
        logits, other_stuff = self.model(x, encoder_padding_mask=encoder_padding_mask)
        return logits
   
    def generate_text(self, start_string, generation_length=100, device='cuda'):
        # Evaluation mode
        self.model.eval()
        self.model.to(device)
 
 
        # Convert start string to numbers
        input_eval = self.tokenizer.stoi(start_string)
        print(input_eval)
        input_eval = torch.tensor(input_eval).unsqueeze(0).to(device)
 
 
        # Empty list to store generated text
        text_generated = []
 
 
        # No gradients needed
        with torch.no_grad():
            for _ in range(generation_length):
                predictions = self.forward(input_eval)

                predictions = predictions[:, -1]
                # Apply softmax to get probabilities
                predictions = F.softmax(predictions, dim=-1)

                # Get the top-k predicted words
                top_k = 5
                topk_values, topk_indices = torch.topk(predictions, k=top_k, dim=-1)
                # Sample from topk_values
                predicted_id = torch.multinomial(topk_values.squeeze(), num_samples=1)
                # Get the corresponding topk_indices
                predicted_id = torch.gather(topk_indices.squeeze(), dim=-1, index=predicted_id)

                # Add predicted word to the input (to be used as next input sequence)
                input_eval = torch.cat([input_eval, predicted_id.unsqueeze(-1)], dim=-1)

                # Convert predicted word id to word
                predicted_word = self.tokenizer.itos(predicted_id.tolist())

                text_generated.append(predicted_word)
 
 
        return start_string + ' ' + ' '.join(text_generated)
    

class SplitformerModel(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            value_embed_dim: int,
            attention_heads: int,
            ffn_dim: int,
            layers: int,
            dropout: float,
            activation_dropout: float,
            vocab_size: int,
            checkpoint_activations: bool,
            fsdp: bool,
            max_seq_len: int,
            tokens_per_pass: int):
        """ Use parameters to create corresponding RetNet model
        Args:
            embed_dim (int): Dimension size of each embedded token.
            value_embed_dim (int): Value embed dimension size.
            attention_heads (int): Number of attention heads in MHA module.
            ffn_dim (int): Hidden layer size of Feed Forward Network (FFN).
            layers (int): Number of retention network layers.
            dropout (float): Probability of an element to be zeroed during dropout.
            activation_dropout (float): Probability of an element to be zeroed
                during dropout after activation between FFN layers.
            vocab_size (int): Maximum vocabulary size (number of unique tokens in
                vocabulary.
            checkpoint_activations (bool): Whether to perform checkpointing or not
                (done with the FairScale library).
            fsdp (bool): Whether to shard Module parameters across data parallel
                workers or not (with the FairScale library).
            max_seq_len (int): Size of context window.
            tokens_per_pass (int): Number of tokens to generate for each forward pass in the splitformer model.
        """
        super().__init__()
 
 
        self.model_params = {
                "embed_dim": embed_dim,
                "value_embed_dim": value_embed_dim,
                "attention_heads": attention_heads,
                "ffn_dim": ffn_dim,
                "layers": layers,
                "dropout": dropout,
                "activation_dropout": activation_dropout,
                "vocab_size": vocab_size,
                "checkpoint_activations": checkpoint_activations,
                "fsdp": fsdp,
                "max_seq_len": max_seq_len,
                "tokens_per_pass": tokens_per_pass
                }
 
 
        config = DecoderConfig(
                decoder_embed_dim=embed_dim,
                decoder_value_embed_dim=value_embed_dim,
                decoder_attention_heads=attention_heads,
                decoder_ffn_embed_dim=ffn_dim,
                decoder_layers=layers,
                dropout=dropout,
                activation_dropout=activation_dropout,
                vocab_size=vocab_size * tokens_per_pass, # This is different
                checkpoint_activations=checkpoint_activations,
                fsdp=fsdp)
 
 
        # Save max_seq_len for padding later
        self.max_seq_len = max_seq_len
 
 
        # Save vocab_size for final dimensions later
        self.vocab_size = vocab_size

        # Save tokens_per_pass for final dimensions later
        self.tokens_per_pass = tokens_per_pass
 
 
        # Create embeddings with index 0 representing padding
        self.text_embeddings = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embed_dim,
                padding_idx=1)
 
 
        self.model = SplitformerDecoder(config, embed_tokens=self.text_embeddings)
 
 
    def forward(self, x: torch.Tensor, encoder_padding_mask=False) -> torch.Tensor:
        logits, other_stuff = self.model(x, encoder_padding_mask=encoder_padding_mask)
        return logits
   
    def generate_text(self, start_string, generation_length=100, device='cuda'):
        # Change this to generate 2 tokens at a time
        # Evaluation mode
        self.model.eval()
        self.model.to(device)
 
 
        # Convert start string to numbers
        input_eval = self.tokenizer.stoi(start_string)
        print(input_eval)
        input_eval = torch.tensor(input_eval).unsqueeze(0).to(device)
 
 
        # Empty list to store generated text
        text_generated = []
 
 
        # No gradients needed
        with torch.no_grad():
            for _ in range(generation_length // self.tokens_per_pass):
                predictions = self.forward(input_eval)
                # Reshape predictions
                C = self.vocab_size
                S = self.tokens_per_pass
                B, T, A = predictions.shape
                # Select the last T tokens
                predictions = predictions[:, -1:, :]
                T = 1
                predictions = predictions.reshape(B * T * S, C)
                # Apply softmax to get probabilities
                predictions = F.softmax(predictions, dim=-1)

                # Add predicted words to the input (to be used as next input sequence)
                predicted_ids = torch.argmax(predictions, dim=1)
 
                # Add predicted word to the input (to be used as next input sequence)
                input_eval = torch.cat([input_eval, predicted_ids.unsqueeze(0)], dim=-1)
 
                # Convert predicted word id to word
                predicted_words = self.tokenizer.itos(predicted_ids.tolist())
 
                text_generated.append(predicted_words)
 
 
        return start_string + ' ' + ' '.join(text_generated)
   
 
 
if __name__ == "__main__":
    # Initialize, setup, and parse the argument parser
    parser = ArgumentParser(
            prog="Model Trainer",
            description="Used to train comparable RetNet, Transformer models.")
 
 
    parser.add_argument("-a", "--activation-dropout", type=float, default=0.0,
            help="Probability of element to be zeroed in dropout layer " + \
                    "after activation between FFN layers.")
    parser.add_argument("-c", "--checkpoint-activations", type=bool,
            default=False, help="Use checkpointing.")
    parser.add_argument("-d", "--dropout", type=float, default=0.0,
            help="Probability of element to be zeroed in dropout layer.")
    parser.add_argument("-e", "--embed-dim", type=int, default=768,
            help="Embedding dimension size of each token.")
    parser.add_argument("-f", "--ffn-dim", type=int, default=1280,
            help="FFN hidden layer size.")
    parser.add_argument("--fsdp", type=bool, default=False,
            help="Module parameters sharded across data parallel workers.")
    parser.add_argument("-l", "--layers", type=int, default=12,
            help="Number of stacked layers in model.")
    parser.add_argument("--lr", type=float, required=True,
            help="Learning rate of model to train.")
    parser.add_argument("-m", "--model", required=True,
            choices=["retnet", "transformer", "splitformer"],
            help="Name of model architecture to train.")
    parser.add_argument("-n", "--heads", type=int, default=3,
            help="Number of heads. Head architecture changes based on model.")
    parser.add_argument("-s", "--seq-len", type=int, default=512,
            help="Sequence length (context window size).")
    parser.add_argument("--value-embed-dim", type=int, default=1280,
            help="Value embed dimension size.")
    parser.add_argument("--vocab-size", type=int, required=True,
            help="Maximum number of unique tokens in vocabulary.")
    parser.add_argument("--batch-size", type=int, default=32,
            help="Batch size.")
    parser.add_argument("--device", type=str, default='cuda',
            help="Device to use (GPU).")
    parser.add_argument("--epochs", type=int, default=10,
            help="Number of epochs to train for.")
    parser.add_argument("--tokens-per-pass", type=int, default=1,
            help="Number of tokens to generate for each forward pass in the splitformer model.")
    parser.add_argument("--speed-test", type=bool, default=False,
            help="Whether to run a speed test or not.")
    parser.add_argument("--save-name", type=str, default="model",
            help="Name to save the model to.")
 
 
    args = parser.parse_args()
   
    # Test that the head dimension will be an even, whole number
    assert args.embed_dim % (args.heads * 2) == 0, \
            "Head Dimension must be even to perform Rotary Position " + \
            f"Embedding ({args.embed_dim} / {args.heads} = " + \
            f"{args.embed_dim / args.heads} -- not an even, whole number)! " + \
            "Try changing the Embedding Dimension or number of heads."
 
 
    # Test that the value embedding dimension is divisible by number of heads
    assert args.value_embed_dim % args.heads == 0, \
            "Value Embed Dimension not divisible by number of heads " + \
            f"({args.value_embed_dim} % {args.heads} != 0)!"
    
    # Test that if it is a splitformer model
    if args.model == "splitformer":
        # That the tokens per pass is less than the sequence length
        assert args.tokens_per_pass < args.seq_len, \
                "Tokens per pass must be less than the sequence length!"
        # And assert tokens per pass is greater than 1
        assert args.tokens_per_pass > 1, \
                "Tokens per pass must be greater than 1!"

 
    # Create requested model
    if args.model == "retnet":
        model = RetNetModel(
                embed_dim=args.embed_dim,
                value_embed_dim=args.value_embed_dim,
                retention_heads=args.heads,
                ffn_dim=args.ffn_dim,
                layers=args.layers,
                dropout=args.dropout,
                activation_dropout=args.activation_dropout,
                vocab_size=args.vocab_size,
                checkpoint_activations=args.checkpoint_activations,
                fsdp=args.fsdp,
                max_seq_len=args.seq_len)
    elif args.model == "transformer":
        model = TransformerModel(
                embed_dim=args.embed_dim,
                value_embed_dim=args.value_embed_dim,
                attention_heads=args.heads,
                ffn_dim=args.ffn_dim,
                layers=args.layers,
                dropout=args.dropout,
                activation_dropout=args.activation_dropout,
                vocab_size=args.vocab_size,
                checkpoint_activations=args.checkpoint_activations,
                fsdp=args.fsdp,
                max_seq_len=args.seq_len)
    elif args.model == "splitformer":
        model = SplitformerModel(
                embed_dim=args.embed_dim,
                value_embed_dim=args.value_embed_dim,
                attention_heads=args.heads,
                ffn_dim=args.ffn_dim,
                layers=args.layers,
                dropout=args.dropout,
                activation_dropout=args.activation_dropout,
                vocab_size=args.vocab_size,
                checkpoint_activations=args.checkpoint_activations,
                fsdp=args.fsdp,
                max_seq_len=args.seq_len,
                tokens_per_pass=args.tokens_per_pass)
        

    # TODO: Allow for different datasets and tokenizers
    # Load the dataset
    train_loader, valid_loader, test_loader, tokenizer = load_wikitext2(max_seq_len=args.seq_len, batch_size=args.batch_size, tokens_per_pass=args.tokens_per_pass)
    model.tokenizer = tokenizer
   
 
 
    # Print all arguments for recordkeeping
    print('Arguments:')
    arg_table = []
    row = []
    for i, arg in enumerate(vars(args)):
        row.append(f'{arg}: {getattr(args, arg)}')
        if (i + 1) % 4 == 0:
            arg_table.append(row)
            row = []
    if row:
        arg_table.append(row)
 
 
    print(tabulate(arg_table, tablefmt="grid"))
 
 
    # Print model info
    print('\nModel Summary:')
    model_summary(model, input_data=torch.ones(1, args.seq_len).long())
 
 
    # Print estimated loss if it hasn't learned anything
    print('\nEstimated Loss if guessing:')
    print(f'-log(1 / {args.vocab_size}) = {-torch.log(torch.tensor(1 / args.vocab_size))}')
 
 
    # Define loss function
    loss_fn = nn.CrossEntropyLoss()
 
 
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
 
 
    # Define the device to use
    device = torch.device(args.device)
 
 
    # Put model on device
    model = model.to(device)
 
 
    if not args.speed_test:
        # Train the model
        print('\nTraining model...')
        start = time.time()
        for epoch in range(args.epochs):
            print(f'Epoch {epoch + 1}')
            for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, mininterval=60)): # Prints progress bar every mininterval seconds
                
    
                # Make a sample padding mask where there are 0's for padding and 1's for real tokens
                encoder_padding_mask = torch.ones(inputs.shape, dtype=torch.bool)
                encoder_padding_mask[inputs == 1] = False
    
                # Put inputs and targets on device
                inputs = inputs.to(device)
                targets = targets.to(device)
                encoder_padding_mask = encoder_padding_mask.to(device)
        
                # Zero out gradients
                optimizer.zero_grad()
        
                # Get model predictions
                predictions = model(inputs, encoder_padding_mask=encoder_padding_mask)

                if args.model != "splitformer": 
                    # Reshape the model outputs to match the expected shape for CrossEntropyLoss
                    B, T, C = predictions.shape
                    predictions = predictions.reshape(B * T, C)
                    B, T = targets.shape
                    targets = targets.reshape(B * T)
                else:
                    # Reshape the model outputs to match the expected shape for CrossEntropyLoss
                    S = args.tokens_per_pass
                    C = args.vocab_size
                    # A == S*C
                    B, T, A = predictions.shape
                    predictions = predictions.reshape(B * T * S, C)
                    B, T, S = targets.shape
                    targets = targets.reshape(B * T * S)
        
                # Calculate loss
                loss = loss_fn(predictions, targets)
        
                # Backpropagate loss
                loss.backward()
        
                # Update parameters
                optimizer.step()
    
                # Run validation n times per epoch
                n = 1
                if batch_idx % (len(train_loader) // n) == 0:
                    # Print train loss
                    print(f"Train Loss: {loss.item()}")
                    model.eval()
                    with torch.no_grad():
                        total_loss = 0
                        total_samples = 0
                        for val_inputs, val_targets in valid_loader:
                            # Put validation inputs and targets on device
                            val_inputs = val_inputs.to(device)
                            val_targets = val_targets.to(device)
                        
                            # Get validation predictions
                            val_predictions = model(val_inputs)
                        
                            if args.model != "splitformer": 
                                # Reshape the model outputs to match the expected shape for CrossEntropyLoss
                                B, T, C = val_predictions.shape
                                val_predictions = val_predictions.reshape(B * T, C)
                                B, T = val_targets.shape
                                val_targets = val_targets.reshape(B * T)
                            else:
                                # Reshape the model outputs to match the expected shape for CrossEntropyLoss
                                S = args.tokens_per_pass
                                C = args.vocab_size
                                # A == S*C
                                B, T, A = val_predictions.shape
                                val_predictions = val_predictions.reshape(B * T * S, C)
                                B, T, S = val_targets.shape
                                val_targets = val_targets.reshape(B * T * S)
                        
                            # Calculate validation loss
                            val_loss = loss_fn(val_predictions, val_targets)
                            total_loss += val_loss.item() * val_inputs.size(0)
                            total_samples += val_inputs.size(0)
                    
                        # Calculate average validation loss
                        avg_val_loss = total_loss / total_samples
                        print(f"Validation Loss: {avg_val_loss}")
                
                    model.train()
        end = time.time()
        print(f"Time to train: {end - start}")
    
    
        # Test the model
        print('\nTesting model...')
        model.eval()
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, mininterval=60): # Prints progress bar every mininterval seconds
                # Put inputs and targets on device
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Get model predictions
                predictions = model(inputs)
            
                if args.model != "splitformer": 
                    # Reshape the model outputs to match the expected shape for CrossEntropyLoss
                    B, T, C = predictions.shape
                    predictions = predictions.reshape(B * T, C)
                    B, T = targets.shape
                    targets = targets.reshape(B * T)
                else:
                    # Go through predictions on the T dimension and select indexes 0, S, 2S, 3S, etc. for perplexity
                    predictions = predictions[:, 0::S, :]
                    # Go through targets on the T dimension and select indexes 0, S, 2S, 3S, etc. for perplexity
                    targets = targets[:, 0::S, :]

                    # Reshape the model outputs to match the expected shape for CrossEntropyLoss
                    S = args.tokens_per_pass
                    C = args.vocab_size
                    # A == S*C
                    B, T, A = predictions.shape
                    predictions = predictions.reshape(B * T * S, C)
                    B, T, S = targets.shape
                    targets = targets.reshape(B * T * S)
            
                # Calculate loss
                loss = loss_fn(predictions, targets)
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
    
        # Calculate average loss
        avg_loss = total_loss / total_samples
        print(f"Test Loss: {avg_loss}")
        # Calculate perplexity
        perplexity = torch.exp(torch.tensor(avg_loss))
        print(f"Perplexity: {perplexity}")
 
 
    # Generate text from the model
    # Print how long it takes to generate the text
    print('\nGenerating text...')
    start = time.time()
    
    print(model.generate_text(start_string="<pad>", generation_length=100, device=device))
    print(model.generate_text(start_string="= valkyria", generation_length=100, device=device))
    print(model.generate_text(start_string="= = reception =", generation_length=100, device=device))
    print(model.generate_text(start_string="the item was intended", generation_length=100, device=device))

    end = time.time()
    print(f"Time to generate text: {end - start}")
 

    # # Save the model
    # if args.save_name == "model":
    #     save_name = args.model
    # else:
    #     save_name = args.save_name
    # save_model(save_name, model)
