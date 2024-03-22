import os
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from GPT2.config import GPT2Config
from GPT2.encoder import get_encoder
from GPT2.model import GPT2LMHeadModel
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

from load_and_save_tokenizer import paths
from nltk.translate.bleu_score import SmoothingFunction
from statistics import mean
import gc

from tokenizer_utils import handel_encode, handel_decode

import sys

class KinyaDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        return input_ids, attention_mask


# encoded = handel_encode("Urukundo ni umutima w'umuntu wese.")
# print(encoded)
# # Check decoder
# print("Decoded:", handel_decode(encoded, skip_special_tokens=True))


# load tokenized kinystory data

# Define the file path
file_path = 'tokenized_data.pt'

# Check if the file exists
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    sys.exit()

# Load the data
all_data = torch.load(file_path)

# Convert the data into a list of dictionaries
all_data_dicts = [{'input_ids': torch.tensor(data[0], dtype=torch.long), 'attention_mask': torch.tensor(data[1], dtype=torch.long)} for data in all_data]

# Create the KinyaDataset
dataset = KinyaDataset(all_data_dicts)
# print the size of the dataset
print(f"Dataset size: {len(dataset)}")

# Split the data into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
config = GPT2Config()
model = GPT2LMHeadModel(config)
model.dropout = nn.Dropout(0.1)  # Add dropout layer
model.to(device)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
model.train()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# Train the model
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
epochs = 10

rouge = Rouge()
import wandb
wandb.login(key="3644f3d76a394594794c1b136a20f75303e871ba")#API Key is in your wandb account, under settings (wandb.ai/settings)
run = wandb.init(
    name = "kinya-story", ## Wandb creates random run names if you skip this field
    #reinit = True, ### Allows reinitalizing runs when you re-run this cell
    id ="kinya-story", ### Insert specific run id here if you want to resume a previous run
    resume = "must", ### You need this to resume previous runs, but comment out reinit = True when using this
    project = "project-ablations", ### Project should be created in your wandb account
    config = config ### Wandb Config for your run
)
# Create a SmoothingFunction object
smoothie = SmoothingFunction().method4


# Define a function to safely get the Rouge-L score
# Define a function to safely get the Rouge-L score
def get_rouge_l_score(score):
    if isinstance(score, dict) and 'rouge-l' in score and isinstance(score['rouge-l'], dict):
        return score['rouge-l']['f']
    else:
        return 0.0  # or any other default value you prefer

if __name__ == '__main__':
    gc.collect() # These commands help you when you face CUDA OOM error
    torch.cuda.empty_cache()
    # Check if there is a checkpoint
    if os.path.exists('gpt2_story_generator.pth'):
        try:
            state_dict = torch.load('gpt2_story_generator.pth', map_location='cpu' if not torch.cuda.is_available() else None)
            model.load_state_dict(state_dict)
        except:
            print('Error loading the model')

    start_epoch = 0
    for epoch in range(start_epoch, epochs):
        gc.collect() # These commands help you when you face CUDA OOM error
        torch.cuda.empty_cache()
        start_epoch = epoch
        # Training loop
        model.train()
        train_perplexities = []
        train_accuracies = []
        train_bleu_scores = []
        train_rouge_scores = []
        curr_lr = float(optimizer.param_groups[0]['lr'])
        train_progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}', leave=False)
        for i, batch in enumerate(train_progress_bar):
            # Check how many elements are in the batch
            #print(len(batch))
            
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Get logits from the model
            logits = model(input_ids=input_ids)[0]

            # Compute the loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate perplexity
            perplexity = torch.exp(loss).item()
            train_perplexities.append(perplexity)

            # Calculate the predictions
            predictions = torch.argmax(logits, dim=-1)

            # Calculate the accuracy
            accuracy = accuracy_score(input_ids.flatten().cpu(), predictions.flatten().cpu())
            train_accuracies.append(accuracy)

            # Calculate BLEU and ROUGE scores
            reference = handel_decode(input_ids[0].cpu().tolist(), skip_special_tokens=True)
            candidate = handel_decode(predictions[0].cpu().tolist(), skip_special_tokens=True)
            bleu_score = sentence_bleu([reference.split()], candidate.split(), smoothing_function=SmoothingFunction().method7)
            try:
                if candidate.strip():  # Check if candidate is not empty
                    rouge_scores = rouge.get_scores(candidate, reference, avg=True)
                else:
                    print("Warning: Empty candidate.")
                    rouge_scores=0
            except Exception as ex:
                rouge_scores=0
                print(f"Exception raised: {ex}")
            average_f1_score_rouge_l = get_rouge_l_score(rouge_scores)
            train_bleu_scores.append(bleu_score)
            train_rouge_scores.append(average_f1_score_rouge_l)

            train_progress_bar.set_postfix({'loss': loss.item(), 'perplexity': perplexity, 'accuracy': accuracy, 'bleu_score': bleu_score, 'rouge_score': average_f1_score_rouge_l})
        train_progress_bar.close()

        # Validation loop
        model.eval()
        val_losses = []
        val_perplexities = []
        val_accuracies = []
        val_bleu_scores = []
        val_rouge_scores = []
        val_progress_bar = tqdm(val_loader, desc=f'Validation Epoch {epoch+1}', leave=False)
        with torch.no_grad():
            for i, batch in enumerate(val_progress_bar):
                input_ids, attention_mask = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                # Get logits from the model
                logits = model(input_ids=input_ids)[0]

                # Compute the loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                val_losses.append(loss.item())
                scheduler.step(loss.item())

                # Calculate perplexity
                perplexity = torch.exp(loss).item()
                val_perplexities.append(perplexity)

                # Calculate the predictions
                predictions = torch.argmax(logits, dim=-1)

                # Calculate the accuracy
                accuracy = accuracy_score(input_ids.flatten().cpu(), predictions.flatten().cpu())
                val_accuracies.append(accuracy)

                # Calculate BLEU and ROUGE scores
                reference = handel_decode(input_ids[0].cpu().tolist(), skip_special_tokens=True)
                candidate = handel_decode(predictions[0].cpu().tolist(), skip_special_tokens=True)
                bleu_score = sentence_bleu([reference.split()], candidate.split(), smoothing_function=SmoothingFunction().method7)
                try:
                    if candidate.strip():  # Check if candidate is not empty
                        rouge_scores = rouge.get_scores(candidate, reference, avg=True)
                    else:
                        print("Warning: Empty candidate.")
                        rouge_scores=0
                except Exception as ex:
                    rouge_scores=0
                    print(f"Exception raised: {ex}")
                    average_f1_score_rouge_l = get_rouge_l_score(rouge_scores)

                val_bleu_scores.append(bleu_score)
                val_rouge_scores.append(average_f1_score_rouge_l)

                val_progress_bar.set_postfix({'loss': loss.item(), 'perplexity': perplexity, 'accuracy': accuracy, 'bleu_score': bleu_score, 'rouge_score': average_f1_score_rouge_l})
        val_progress_bar.close()

        # Calculate average metrics for training and validation
        avg_train_perplexity = sum(train_perplexities) / len(train_perplexities)
        avg_train_accuracy = sum(train_accuracies) / len(train_accuracies)
        avg_train_bleu = sum(train_bleu_scores) / len(train_bleu_scores)
        avg_train_rouge = sum(train_rouge_scores) / len(train_rouge_scores)
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_perplexity = sum(val_perplexities) / len(val_perplexities)
        avg_val_accuracy = sum(val_accuracies) / len(val_accuracies)
        avg_val_bleu = sum(val_bleu_scores) / len(val_bleu_scores)
        avg_val_rouge = sum(val_rouge_scores) / len(val_rouge_scores)

        wandb.log({
            'lr'        : curr_lr,
            'avg_train_perplexity': avg_train_perplexity,
            'avg_train_accuracy': avg_train_accuracy,
            'avg_train_bleu': avg_train_bleu,
            'avg_train_rouge': avg_train_rouge,
            'avg_val_loss': avg_val_loss,
            'avg_val_perplexity': avg_val_perplexity,
            'avg_val_accuracy': avg_val_accuracy,
            'avg_val_bleu': avg_val_bleu,
            'avg_val_rouge': avg_val_rouge
        })

        # Print or log the metrics
        print(f"Epoch: {epoch+1}, Train Loss: {loss.item()}, Train Perplexity: {avg_train_perplexity}, Train Accuracy: {avg_train_accuracy}, Train BLEU: {avg_train_bleu}, Train ROUGE: {avg_train_rouge}")
        print(f"Epoch: {epoch+1}, Validation Loss: {avg_val_loss}, Validation Perplexity: {avg_val_perplexity}, Validation Accuracy: {avg_val_accuracy}, Validation BLEU: {avg_val_bleu}, Validation ROUGE: {avg_val_rouge}")

        # Save the model
        torch.save(model.state_dict(), 'gpt2_story_generator.pth')
        wandb.save('gpt2_story_generator.pth')
