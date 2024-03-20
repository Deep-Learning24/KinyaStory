from pathlib import Path
from Tokenizer import BPE_token
import os

# The folder 'text' contains all the files
paths = [str(x) for x in Path("./rocStoriesData/").glob("**/*.csv")]
print(paths)
tokenizer = BPE_token()
# Train the tokenizer model
tokenizer.bpe_train(paths)

# Saving the tokenized data in the project directory
save_dir = 'tokenized_data'
save_path = os.path.join(os.getcwd(), save_dir)  # Get the full path to the save directory
tokenizer.save_tokenizer(save_path)
