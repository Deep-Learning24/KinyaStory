import os
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from GPT2.config import GPT2Config
from GPT2.model import GPT2LMHeadModel
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

from nltk.translate.bleu_score import SmoothingFunction
from torch.nn.utils.rnn import pad_sequence
from statistics import mean

from tokenizer_utils import handel_encode, handel_decode
import wandb
import sys
import h5py
from torch.cuda.amp import autocast

sys.path.append('../')


class DataPreparator:
    def __init__(self, tokenizer, max_length=128):

        self.tokenizer = tokenizer
        self.max_length = max_length

    def prepare_datasets(self, text_files_path, train_csv_path, test_csv_path, output_dir):
        """
        Prepares and saves tokenized datasets into separate HDF5 files for training, validation, and testing.
        Args:
            text_files_path (str): Directory containing text files for training.
            train_csv_path (str): Path to the CSV file for training.
            test_csv_path (str): Path to the CSV file to be split for validation and testing.
            output_dir (str): Directory where the HDF5 files will be saved.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Tokenize and save training data
        train_output_path = os.path.join(output_dir, "train_dataset.hdf5")
        self._tokenize_and_save([text_files_path, train_csv_path], train_output_path, dataset_type="train")

        # Tokenize and split test data for validation and testing
        test_df = pd.read_csv(test_csv_path)
        # Splitting the DataFrame for validation and test sets
        val_df = test_df.sample(frac=0.5, random_state=42)
        test_df.drop(val_df.index, inplace=True)

        val_output_path = os.path.join(output_dir, "val_dataset.hdf5")
        test_output_path = os.path.join(output_dir, "test_dataset.hdf5")
        self._tokenize_and_save_df(val_df, val_output_path, dataset_type="validation")
        self._tokenize_and_save_df(test_df, test_output_path, dataset_type="test")

    def _tokenize_and_save(self, file_paths, output_path, dataset_type):
        with h5py.File(output_path, 'w') as hf:
            hf.create_dataset("input_ids", (0, self.max_length), maxshape=(None, self.max_length), dtype='i8')
            hf.create_dataset("attention_mask", (0, self.max_length), maxshape=(None, self.max_length), dtype='i8')
            total_entries = 0

            for file_path in file_paths:
                if os.path.isdir(file_path):  # Directory of text files
                    for text_file in os.listdir(file_path):
                        full_path = os.path.join(file_path, text_file)
                        self._process_file(full_path, hf)
                else:  # Single CSV file
                    df = pd.read_csv(file_path)
                    self._tokenize_and_save_df(df, output_path, dataset_type)

            print(f"Total entries processed and saved for {dataset_type}: {total_entries}")

    def _tokenize_and_save_df(self, df, output_path, dataset_type):
        with h5py.File(output_path, 'a') as hf:  # Ensure appending mode
            for _, row in tqdm(df.iterrows(), desc=f"Tokenizing {dataset_type}"):
                text = row['content']  # Assuming 'content' column contains text to tokenize
                self._tokenize_and_append(text, hf)

    def _tokenize_and_append(self, text, hf):
        input_ids, attention_mask = handel_encode(text)
        current_size = hf["input_ids"].shape[0]
        hf["input_ids"].resize((current_size + 1, self.max_length))
        hf["attention_mask"].resize((current_size + 1, self.max_length))
        hf["input_ids"][current_size, :] = input_ids[:self.max_length]
        hf["attention_mask"][current_size, :] = attention_mask[:self.max_length]

    def _process_file(self, file_path, hf):
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    self._tokenize_and_append(line.strip(), hf)



class PretrainDataset(Dataset):
    def __init__(self, hdf5_file_path):
        """
        Initializes the dataset from an HDF5 file containing tokenized data.

        Args:
            hdf5_file_path (str): Path to the HDF5 file.
        """
        self.hdf5_file_path = hdf5_file_path
        with h5py.File(self.hdf5_file_path, 'r') as hf:
            self.length = hf['input_ids'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file_path, 'r') as hf:
            input_ids = torch.tensor(hf['input_ids'][idx], dtype=torch.long)
            attention_mask = torch.tensor(hf['attention_mask'][idx], dtype=torch.long)
            # Assuming you want to ignore padding in the loss calculation
            labels = torch.where(input_ids == 0, torch.tensor(-100), input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }



def collate_fn(batch):
    """
    Collates batch data, dynamically padding to the longest sequence in the batch.
    It also prepares 'labels' tensor, which matches 'input_ids' but with padding tokens set to -100.

    Args:
        batch: A list of dictionaries with 'input_ids', 'attention_mask', and 'labels'.

    Returns:
        A dictionary with batched 'input_ids', 'attention_mask', and 'labels', all padded to the same length.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Pad sequences
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        'input_ids': input_ids_padded.to(device),
        'attention_mask': attention_masks_padded.to(device),
        'labels': labels_padded.to(device)
    }


def get_rouge_l_score(score):
    if isinstance(score, dict) and 'rouge-l' in score and isinstance(score['rouge-l'], dict):
        return score['rouge-l']['f']
    else:
        return 0.0  # or any other default value you prefer


class Pretrain:
    def __init__(self, model_path, tokenizer, device='cuda'):

        self.model_path = model_path

        self.tokenizer = tokenizer

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.config = GPT2Config()

        self.model = GPT2LMHeadModel(self.config)

        self.model.to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)

        wandb.login(key="3644f3d76a394594794c1b136a20f75303e871ba")#API Key is in your wandb account, under settings (wandb.ai/settings)
        
        wandb.init(
            project="project-ablations", 
            config=self.config.to_dict(),
            name = "kinya-story-pretrain", ## Wandb creates random run names if you skip this field
            #reinit = True, ### Allows reinitalizing runs when you re-run this cell
            id ="kinya-story-pretrain", ### Insert specific run id here if you want to resume a previous run
            resume = "must", ### You need this to resume previous runs, but comment out reinit = True when using this
            )

        # Create a SmoothingFunction object
        self.smoothie = SmoothingFunction().method4

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3)
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.rouge = Rouge()
        self.load_model()

    def train(self, train_loader, val_loader, epochs=30):

        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch in progress_bar:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with autocast():
                    loss = self.model(**batch)
                    
                total_loss += loss.item()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                progress_bar.set_postfix({'Training Loss': f'{loss.item():.4f}'})
                
            avg_train_loss = total_loss / len(train_loader)
            eval_loss, avg_bleu, avg_rouge, avg_perplexity = self.evaluate(val_loader)
            
            wandb.log({"avg_train_loss": avg_train_loss, "eval_loss": eval_loss, "avg_bleu": avg_bleu, "avg_rouge": avg_rouge, "avg_perplexity": avg_perplexity,"epoch": epoch+1})

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        bleu_scores = []
        rouge_scores = []
        perplexities = []
        with torch.no_grad():
            for batch in tqdm(loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                # get the labels which are not equal to -100
                labels = labels[labels != -100]
                with autocast():
                    logits, _ = self.model(input_ids, attention_mask=attention_mask)
                    loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                  
                predicted_ids = torch.argmax(logits, dim=-1)
                bleu_score = self.calculate_bleu_score(labels, predicted_ids)
                rouge_score = self.calculate_rouge_score(labels, predicted_ids)
                perplexity = torch.exp(loss)
                perplexities.append(perplexity.item())
                bleu_scores.append(bleu_score)
                rouge_scores.append(rouge_score)
                total_loss += loss.item()
        avg_bleu = mean(bleu_scores)
        avg_rouge = mean(rouge_scores)
        avg_loss = total_loss / len(loader)
        return avg_loss, avg_bleu, avg_rouge, mean(perplexities)

    
    def calculate_bleu_score(self, labels, predicted_ids):
        bleu_scores = []
        for i in range(len(labels)):
            # Exclude -100 values from labels before decoding
            label = labels[i]
            proper_label = []
                # Decode label tensor
            for token in label:
                if token>0:
                    proper_label.append(token)
                        
            decoded_label = handel_decode(proper_label)
            predicted = handel_decode(predicted_ids[i])
            bleu_score = sentence_bleu(decoded_label, predicted, smoothing_function=self.smoothie)
            bleu_scores.append(bleu_score)
        return mean(bleu_scores)
    

    def calculate_rouge_score(self, labels, predicted_ids):
        rouge_scores = []
        for i in range(len(labels)):
            # Exclude -100 values from labels before decoding
            label = labels[i]
            proper_label = []
                # Decode label tensor
            for token in label:
                if token>0:
                    proper_label.append(token)
            decoded_label = handel_decode(proper_label)
            predicted = handel_decode(predicted_ids[i])
            rouge_score = self.rouge.get_scores(predicted, decoded_label)
            rouge_scores.append(get_rouge_l_score(rouge_score))
        return mean(rouge_scores)
    
    
    def save_model(self, filename="best_gpt2_model.pt"):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        torch.save(self.model.state_dict(), os.path.join(self.model_path, filename))
        print(f'Model saved to {os.path.join(self.model_path, filename)}')

    def load_model(self, filename="best_gpt2_model.pt"):
        model_file = os.path.join(self.model_path, filename)
        if os.path.exists(model_file):
            self.model.load_state_dict(torch.load(model_file))
            self.model.eval()
            print(f'Model loaded from {model_file}')
        else:
            print("Model file not found.")


def main():
    
    tokenizer = handel_encode
    max_length = 128  # or another value based on your model's capabilities
    
    # Paths to your raw data
    text_files_path = 'Kinyarwanda_Data/Kinyarwanda_Data'
    train_csv_path = 'kinyarwanda news/train.csv'
    test_csv_path = 'kinyarwanda news/test.csv'
    
    # Output directory for processed data
    output_dir = 'pretrain_tokenized_data'
    
    # Initialize and run your data preparation
    data_preparator = DataPreparator(tokenizer=tokenizer, max_length=max_length)
    data_preparator.prepare_datasets(text_files_path, train_csv_path, test_csv_path, output_dir)
    
    # Assuming the above method saves three HDF5 files: train_dataset.hdf5, val_dataset.hdf5, test_dataset.hdf5
    
    # Initialize DataLoader for each dataset
    train_dataset = PretrainDataset(os.path.join(output_dir, "train_dataset.hdf5"))
    val_dataset = PretrainDataset(os.path.join(output_dir, "val_dataset.hdf5"))
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    # Model path where checkpoints will be saved
    model_path = 'models'
    
    # Initialize the training process
    pretrain_instance = Pretrain(
        model_path=model_path,
        tokenizer=tokenizer,  # Note: If your tokenizer needs to be used within Pretrain, ensure it's correctly passed and utilized
        device='cuda'  # or 'cpu'
    )
    
    # Start training
    pretrain_instance.train(train_loader, val_loader, epochs=50)
    
    # Save the final model
    pretrain_instance.save_model()

if __name__ == "__main__":
    main()