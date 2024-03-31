import json
from transformers import AutoTokenizer
import torch
import os
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import gc
import psutil
import glob

class KinyaTokenizerPretrain(object):
    def __init__(self, text_files_path, train_csv_path, test_csv_path):
        self.tokenizer = AutoTokenizer.from_pretrained("jean-paul/KinyaBERT-large", max_length=512)
        self.text_files_path = text_files_path
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path


    def load_text_files(self):
        text_files = glob.glob(os.path.join(self.text_files_path, "*.txt"))
        text_data = []
        for file in text_files:
            with open(file, 'r', encoding='utf-8') as f:
                text_data.append(f.read())
        return text_data

    def load_csv_files(self):
        train_df = pd.read_csv(self.train_csv_path)
        test_df = pd.read_csv(self.test_csv_path)
        return train_df, test_df
        
    def tokenize_file(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
            encoding = self.tokenizer(text, return_tensors='pt')
            return encoding

    def tokenize_text_files(self):
        text_files = glob.glob(os.path.join(self.text_files_path, "*.txt"))
        print(f"Tokenizing {len(text_files)} text files: {text_files}")
        print(f"Memory usage before tokenization: {psutil.virtual_memory().percent}%")
        with Pool() as p:
            for tokenized_data in tqdm(p.imap(self.tokenize_file, text_files), total=len(text_files)):
                yield tokenized_data

    def tokenize_row(self, row):
        story_input = row['title']
        story_output = row['content']
        input_encoding = self.tokenizer(story_input, return_tensors='pt')
        output_encoding = self.tokenizer(story_output, return_tensors='pt')
        return (input_encoding, output_encoding)

    def tokenize_csv_dataset(self):
        train_df, test_df = self.load_csv_files()
        for df in [train_df, test_df]:
            with Pool() as p:
                for tokenized_data in tqdm(p.imap(self.tokenize_row, df.iterrows()), total=len(df)):
                    yield tokenized_data

    def save_tokenized_data(self, tokenized_data, save_path):
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_name = f"{len(os.listdir(save_path))}.pt"
            torch.save(tokenized_data, os.path.join(save_path, file_name))
            print(f"Tokenized data saved to {os.path.join(save_path, file_name)}")

    def merge_tokenized_data(self, save_path):
        merged_data = []
        for file_name in os.listdir(save_path):
            file_path = os.path.join(save_path, file_name)
            tokenized_data = torch.load(file_path)
            merged_data.extend(tokenized_data)
            os.remove(file_path)  # delete the file after loading to free up disk space
        torch.save(merged_data, f"{save_path}.pt")
        print(f"Merged tokenized data saved to {save_path}.pt")

if __name__ == "__main__":
    text_files_path = 'Kinyarwanda_Data/Kinyarwanda_Data'
    train_csv_path = 'kinyarwanda news/train.csv'
    test_csv_path = 'kinyarwanda news/test.csv'
    save_path = 'pretrain_tokenized_data'

    tokenizer = KinyaTokenizerPretrain(text_files_path, train_csv_path, test_csv_path)
    
    for tokenized_data in tokenizer.tokenize_text_files():
        print(f"Memory usage after text files tokenization: {psutil.virtual_memory().percent}%")
        tokenizer.save_tokenized_data(tokenized_data, save_path)
        del tokenized_data
        gc.collect()

    for tokenized_data in tokenizer.tokenize_csv_dataset():
        print(f"Memory usage after CSV data tokenization: {psutil.virtual_memory().percent}%")
        tokenizer.save_tokenized_data(tokenized_data, save_path)
        del tokenized_data
        gc.collect()

    tokenizer.merge_tokenized_data(save_path)
    print("Tokenization complete!")