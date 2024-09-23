# from pathlib import Path
from transformers import GPT2Tokenizer
import torch
import pandas as pd

def get_config():
    return {
        "num_epochs": 1,
        "interval": 100,
        "lr": 1e-4,
        "length": 512,
        "embed_dim": 512,
        "num_blocks": 6, 
        "num_heads": 8, 
        "ff_dim": 2048, 
        # dropout_rate=0.1,
        "batch_size": 1,
        "model_folder": "saved/2/",
        # "dataset_path": "datasets/shakespeare.txt"
        "dataset_path": "datasets\wmt14_translate_de-en_train.csv"
    }

def get_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=True)
    num_added_tokens = tokenizer.add_tokens(['[START]', '[END]'], special_tokens=True)
    # num_added_tokens += tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    vocab_size = tokenizer.vocab_size + num_added_tokens
    return tokenizer, vocab_size

def get_preprocessed_sequence(tokenized_text, step, length, device):
    if step*length + length >= tokenized_text.shape[0]:
        sequence = tokenized_text[step*length:tokenized_text.shape[0]]
    else:
        sequence = tokenized_text[step*length:step*length + length]
    return sequence.to(device)

def get_causal_mask(shape, device):
    causal_mask = torch.triu(torch.ones((shape), dtype=torch.float), diagonal=1).to(device) # make this into a helper function
    causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf')).unsqueeze(0).unsqueeze(0) # change what the thing mask is in the model code with squeezes
    return causal_mask

def get_text():
    file = open('datasets/shakespeare.txt', 'r')
    text = file.read()
    return "[START]" + text + "[END]"

def get_tokenized_text(tokenizer, device):
    tokenized_text = torch.tensor(tokenizer(get_text())['input_ids']).to(device)
    return tokenized_text

def checkpoint(step, interval, model, model_folder, losses):
    avg_interval_loss = sum(losses[-interval:]) / interval
    print('step:', step)
    print('avg loss:', avg_interval_loss)
    print('\n')
    torch.save(model.state_dict(), model_folder + 'model.pth') 

    df_dict = {'losses': losses}
    df = pd.DataFrame(df_dict)
    df.to_csv(model_folder + 'losses.csv', index=False)

def get_preprocessed_sequence_pd_csv(step, train_df, tokenizer, device):
    sequence = train_df.iloc[step]['en']
    sequence = '[START]' + sequence + '[END]'
    tokenized_sequence = tokenizer(sequence, max_length = 1024, truncation = True)['input_ids']
    return torch.tensor(tokenized_sequence).to(device)