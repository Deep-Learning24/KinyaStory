

# class PretrainDataset(Dataset):
#     def __init__(self,text_files_path, train_csv_path, test_csv_path, tokenizer, is_train=True,done_tokeinizin=True):
#         self.tokenizer = tokenizer
#         self.text_files_path = text_files_path
#         self.pretrain_tokenized_data_dir = 'pretrain_tokenized_data'
#         self.train_csv_path = train_csv_path
#         self.test_csv_path = test_csv_path
#         self.is_train = is_train
#         self.done_tokeinizin = done_tokeinizin
#         self.train_tokenized_data_file = os.path.join(self.pretrain_tokenized_data_dir, 'train_tokenized_data.txt')
#         self.test_tokenized_data_file = os.path.join(self.pretrain_tokenized_data_dir, 'test_tokenized_data.txt')
#         self.tokenized_text_data_file = os.path.join(self.pretrain_tokenized_data_dir, 'text_tokenized_data.txt')
#         if not os.path.exists(self.pretrain_tokenized_data_dir):
#             os.makedirs(self.pretrain_tokenized_data_dir)
#         if not self.done_tokeinizin:
#             self.tokenize_text_files()
#             self.tokenize_csv_dataset()
           
#         self.data_length = self.get_data_length()

#     def get_data_length(self):
#         if self.is_train:
#             with open(self.train_tokenized_data_file, 'r') as f1, open(self.tokenized_text_data_file, 'r') as f2:
#                 for i, _ in enumerate(itertools.chain(f1, f2), 1): pass
#         else:
#             with open(self.test_tokenized_data_file, 'r') as f:
#                 for i, _ in enumerate(f, 1): pass
#         return i

#     def tokenize_text_files(self):
#         text_files = glob.glob(os.path.join(self.text_files_path, "*.txt"))
#         print(f"Tokenizing {len(text_files)} text files: {text_files}")
#         print(f"Memory usage before tokenization of text files: {psutil.virtual_memory().percent}%")
#         with open(self.tokenized_text_data_file, 'a') as out:  # Use 'a' mode
#             for text_file in tqdm(text_files, total=len(text_files)):
#                 self.tokenize_file(text_file, out)
#         print("Finished tokenizing text files")

#     def tokenize_csv_dataset(self):
#         train_df = pd.read_csv(self.train_csv_path)
#         test_df = pd.read_csv(self.test_csv_path)
        

#         print(f"Tokenizing csv datasets: {self.train_csv_path}, {self.test_csv_path}")
#         print(f"Memory usage before tokenization of csv datasets: {psutil.virtual_memory().percent}%")
#         for df, tokenized_data_file in zip([train_df, test_df], [self.train_tokenized_data_file, self.test_tokenized_data_file]):
#             with open(tokenized_data_file, 'a') as out:  # Use 'a' mode
#                 for _, row in tqdm(df.iterrows(), total=len(df)):
#                     story_input = row['title']
#                     story_output = row['content']
#                     try:
#                         # Combine the input and output sequences separated by a special token <pad>
#                         combined_text = story_input + '<pad>' + story_output
#                         combined_encoding = self.tokenizer(combined_text)
#                         out.write(str(combined_encoding) + '\n')
#                     except Exception as e:
#                         print(f"Error tokenizing row: {e}")
#                         continue
                   
#         print("Finished tokenizing csv datasets")

#     def tokenize_file(self, file, out):
#         with open(file, 'r', encoding='utf-8') as f:
#             for line in f:
#                 encoding = self.tokenizer(line)
#                 out.write(str(encoding) + '\n')

#     def __len__(self):
#         return self.data_length


#     def __getitem__(self, idx):
#         if self.is_train:
#             with open(self.train_tokenized_data_file, 'r') as f1, open(self.tokenized_text_data_file, 'r') as f2:
#                 for i, line in enumerate(itertools.chain(f1, f2)):
#                     if i == idx:
#                         processed_line = self.process_line(line, idx)
#                         if processed_line is None:
#                             return self.__getitem__(idx + 1)  # Skip to the next line
#                         return processed_line
#         else:
#             with open(self.test_tokenized_data_file, 'r') as f:
#                 for i, line in enumerate(f):
#                     if i == idx:
#                         processed_line = self.process_line(line, idx)
#                         if processed_line is None:
#                             return self.__getitem__(idx + 1)  # Skip to the next line
#                         return processed_line

#     def process_line(self, line, idx):
#         try:
#             input_data = ast.literal_eval(line)
#             if len(input_data) < 2:
#                 print(f"Skipping line {idx}: Incorrect number of sequences")
#                 return None
#             #Input ids are at the even index, while attention mask is at the odd index
#             input_ids = input_data[0]
#             attention_mask = input_data[1]
            
#             assert len(input_ids) == len(attention_mask), f"Mismatched lengths for line {idx}"
#         except (ValueError, SyntaxError, AssertionError) as e:
#             print(f"Skipping line {idx}: {e}")
#             return None

#         input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
#         attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
#         labels_tensor = torch.tensor([token_id if token_id != 0 else -100 for token_id in input_ids], dtype=torch.long)

#         return {
#             'input_ids': input_ids_tensor,
#             'attention_mask': attention_mask_tensor,
#             'labels': labels_tensor
#         }

    

# def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
#     """
#     Collates batch data returned by CustomDataset's __getitem__, padding sequences
#     to ensure consistent length within the batch.

#     Args:
#         batch: List of dictionaries containing 'input_ids', 'attention_mask', and 'labels'.

#     Returns:
#         Dictionary of batched and padded 'input_ids', 'attention_masks', and 'labels'.
#     """
#     # Extract sequences
#     input_ids = [item['input_ids'] for item in batch]
#     attention_masks = [item['attention_mask'] for item in batch]
#     labels = [item['labels'] for item in batch]

#     # Padding
#     input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
#     attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
#     labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

#     # Check if CUDA is available and move the tensors to GPU if it is
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     input_ids_padded = input_ids_padded.to(device)
#     attention_masks_padded = attention_masks_padded.to(device)
#     labels_padded = labels_padded.to(device)

#     return {
#         'input_ids': input_ids_padded,
#         'attention_mask': attention_masks_padded,
#         'labels': labels_padded
#     }

class Pretrain:
    def __init__(self, text_files_path, train_csv_path, test_csv_path, model_path,tokenizer):
        self.text_files_path = text_files_path
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.model_path = model_path
        
        self.config = GPT2Config()
        
        wandb.login(key="3644f3d76a394594794c1b136a20f75303e871ba")#API Key is in your wandb account, under settings (wandb.ai/settings)
        
        self.run = wandb.init(
            name = "kinya-story-pretrain", ## Wandb creates random run names if you skip this field
            #reinit = True, ### Allows reinitalizing runs when you re-run this cell
            id ="kinya-story-pretrain", ### Insert specific run id here if you want to resume a previous run
            resume = "must", ### You need this to resume previous runs, but comment out reinit = True when using this
            project = "project-ablations", ### Project should be created in your wandb account
            config = self.config ### Wandb Config for your run
        )

        # Create a SmoothingFunction object
        self.smoothie = SmoothingFunction().method4
        self.tokenizer = tokenizer
        self.model = GPT2LMHeadModel(self.config)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3)
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.train_dataset = PretrainDataset(self.text_files_path, self.train_csv_path, self.test_csv_path, self.tokenizer)
        self.train_loader = DataLoader(self.train_dataset, batch_size=16, shuffle=True,collate_fn=collate_fn)
        self.test_dataset = PretrainDataset(self.text_files_path, self.train_csv_path, self.test_csv_path, self.tokenizer, is_train=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=16, shuffle=False,collate_fn=collate_fn)
        self.rouge = Rouge()
        self.load_model()

    def train(self, epochs=1):
        best_loss = float('inf')
        for epoch in range(epochs):
            self.model.train()
            losses = []
            total_loss = 0.0
            progress_bar = tqdm(self.train_loader, desc='Training')
            for i, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                loss = self.model(input_ids, attention_mask=attention_mask, lm_labels=labels)
                total_loss += loss.item()
                losses.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                progress_bar.set_description(f"Training (loss {total_loss/(i+1):.4f}) at epoch {epoch+1}/{epochs}")

            eval_loss = self.evaluate()
            self.scheduler.step(eval_loss)
            if eval_loss < best_loss:
                best_loss = eval_loss
                self.save_model()

    def evaluate(self):

        self.model.eval()
        bleu_scores = []
        rouge_scores = []
        perplexities = []
        total_loss = 0

        with tqdm(total=len(self.test_loader), desc="Evaluating", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            for i, batch in enumerate(self.test_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                logits, _ = self.model(input_ids, attention_mask=attention_mask)
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                self.scheduler.step(loss.item())
                predicted_ids = torch.argmax(logits, dim=-1)
                bleu_score = self.calculate_bleu_score(labels, predicted_ids)
                rouge_score = self.calculate_rouge_score(labels, predicted_ids)
                perplexity = torch.exp(loss)
                perplexities.append(perplexity.item())
                bleu_scores.append(bleu_score)
                rouge_scores.append(rouge_score)
                total_loss += loss.item()
                pbar.update(i+1)
            pbar.set_postfix({"BLEU Score": mean(bleu_scores), "ROUGE Score": mean(rouge_scores),"perplexity":mean(perplexities)})
            pbar.refresh()
        avg_loss = total_loss / len(self.test_loader)
        print(f"BLEU Score: {mean(bleu_scores)}")
        print(f"ROUGE Score: {mean(rouge_scores)}")
        print(f"Perplexity: {mean(perplexities)}")
        print(f"Average Loss: {avg_loss}")
        
        self.run.log({"BLEU Score": mean(bleu_scores), "ROUGE Score": mean(rouge_scores),"perplexity":mean(perplexities)}, step=self.run.step)
        return avg_loss
    
    def calculate_bleu_score(self, labels, predicted_ids):
        bleu_scores = []
        for i in range(len(labels)):
            label = handel_decode(labels[i])
            predicted = handel_decode(predicted_ids[i])
            bleu_score = sentence_bleu(label, predicted, smoothing_function=self.smoothie)
            bleu_scores.append(bleu_score)
        return mean(bleu_scores)
    
    
    def calculate_rouge_score(self, labels, predicted_ids):
        rouge_scores = []
        for i in range(len(labels)):
            label = handel_decode(labels[i])
            predicted = handel_decode(predicted_ids[i])
            rouge_score = self.rouge.get_scores(predicted, label)
            rouge_scores.append(get_rouge_l_score(rouge_score))
        return mean(rouge_scores)
    
    def save_model(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        model_file = os.path.join(self.model_path, f"best_gpt2_model.pt")
        torch.save(self.model.state_dict(), model_file)
        print(f"Model saved to {model_file}")
    
    def load_model(self):
        best_model_file = os.path.join(self.model_path, 'best_gpt2_model.pt')
        if os.path.exists(best_model_file):
            self.model.load_state_dict(torch.load(best_model_file))
            self.model.eval()
            print(f"Model loaded from {best_model_file}")
        else:
            print("No model found")
