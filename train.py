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
class GPT2Dataset(Dataset):
    def __init__(self, directory, tokenizer, max_length, exclude_files=None, is_val=False, val_file=None):
        self.input_ids = []
        self.max_length = max_length
        self.tokenizer = tokenizer

        if is_val:
            val_data = pd.read_csv(val_file, delimiter=',', header=None).iloc[:, 1:]
            val_data = val_data.fillna('')
            for _, row in val_data.iterrows():
                row = [str(val) for val in row]
                story = ' '.join(row)
                # Encode story
                story_tokens = self.encode_story(story)  # Pass max_length argument from class attribute
                self.input_ids.append(story_tokens)
        else:
            for filename in os.listdir(directory):
                if exclude_files and filename in exclude_files:
                    continue
                if filename.endswith(".csv"):
                    data = pd.read_csv(os.path.join(directory, filename), delimiter=',', header=None)
                    if filename != "Story_Cloze_Test_Winter_2018_val.csv":
                        data = data.iloc[:, 1:]
                    else:
                        data = data.iloc[:, 1:-1]
                    data = data.fillna('')
                    for _, row in data.iterrows():
                        row = [str(val) for val in row]
                        story = ' '.join(row)
                        story_tokens = self.encode_story(story)  # Pass max_length argument from class attribute
                        self.input_ids.append(story_tokens)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        attention_mask = (input_ids != self.tokenizer.encoder['<PAD>']).long()  # Create attention mask
        return input_ids, attention_mask

    def encode_story(self, story):
        # Encode story and truncate or pad to max_length
        story_tokens = self.tokenizer.encode(story, max_length=self.max_length)  # Pass max_length argument
        story_tokens = story_tokens[:self.max_length] + [self.tokenizer.encoder['<PAD>']] * (self.max_length - len(story_tokens))
        return torch.tensor(story_tokens)






# Load the ROCStories dataset
data_file = 'rocStoriesData/Story_Cloze_Test_Winter_2018_val.csv'
data = pd.read_csv(data_file, delimiter=',', header=None)
texts = data.apply(lambda row: ' '.join(row), axis=1)

# Load tokenizer
enc = get_encoder()  # Assuming get_encoder returns the tokenizer
tokenizer = enc
# Determine the maximum length of stories
# max_len = max(len(enc.encode(story)) for story in texts)
max_len = 512
print("Max length: ", max_len)
data_directory = 'rocStoriesData'
exclude_files = ['Story_Cloze_Test_Spring_2016_val.csv','Story_Cloze_Test_Spring_2016_test.csv','Story_Cloze_Test_Winter_2018_test.csv','ROCStories_spring_2016.csv']
dataset = GPT2Dataset(data_directory, enc, max_len, exclude_files)
val_file = 'rocStoriesData/Story_Cloze_Test_Winter_2018_test.csv'
val_dataset = GPT2Dataset(data_directory, enc, max_len, is_val=True, val_file=val_file)

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

# Train the model
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
epochs = 10

rouge = Rouge()
import wandb
wandb.login(key="3644f3d76a394594794c1b136a20f75303e871ba")#API Key is in your wandb account, under settings (wandb.ai/settings)
run = wandb.init(
    name = "kinya-story baseline model", ## Wandb creates random run names if you skip this field
    reinit = True, ### Allows reinitalizing runs when you re-run this cell
    id ="kinyabase", ### Insert specific run id here if you want to resume a previous run
    #resume = "must", ### You need this to resume previous runs, but comment out reinit = True when using this
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

    start_epoch = 2
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
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Get logits from the model
            logits = model(input_ids=input_ids, attention_mask=attention_mask)[0]

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
            reference = tokenizer.decode(input_ids[0].cpu().tolist(), skip_special_tokens=True)
            candidate = tokenizer.decode(predictions[0].cpu().tolist(), skip_special_tokens=True)
            bleu_score = sentence_bleu([reference.split()], candidate.split(), smoothing_function=SmoothingFunction().method7)
            rouge_scores = rouge.get_scores(candidate, reference, avg=True)
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
                logits = model(input_ids=input_ids, attention_mask=attention_mask)[0]

                # Compute the loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                val_losses.append(loss.item())

                # Calculate perplexity
                perplexity = torch.exp(loss).item()
                val_perplexities.append(perplexity)

                # Calculate the predictions
                predictions = torch.argmax(logits, dim=-1)

                # Calculate the accuracy
                accuracy = accuracy_score(input_ids.flatten().cpu(), predictions.flatten().cpu())
                val_accuracies.append(accuracy)

                # Calculate BLEU and ROUGE scores
                reference = tokenizer.decode(input_ids[0].cpu().tolist(), skip_special_tokens=True)
                candidate = tokenizer.decode(predictions[0].cpu().tolist(), skip_special_tokens=True)
                bleu_score = sentence_bleu([reference.split()], candidate.split(), smoothing_function=SmoothingFunction().method7)
                rouge_scores = rouge.get_scores(candidate, reference, avg=True)
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
