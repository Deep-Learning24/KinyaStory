from KinyaTokenizer import encode, decode
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('./kinyatokenizer')
print(f"Vocab size: {tokenizer.vocab_size}")
print(tokenizer.tokenize("Urukundo ni umutima w'umuntu wese."))

def handel_encode(text):
    return encode(tokenizer,text)
def handel_decode(encoded,skip_special_tokens=True):
  
    return decode(tokenizer,encoded,skip_special_tokens)