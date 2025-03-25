from transformers import GPT2Tokenizer
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def get_config():
    return {
        "length": 512,
        "embed_dim": 512, # 768
        "num_blocks": 6,  # 12
        "num_heads": 8,  # 8
        "ff_dim": 2048, 
        "dropout_rate": 0.1,
        "batch_size": 1,
        "num_epochs": 1,
        "interval": 1000,
        "lr": 1e-4,
        "entropy_weight": 0.01,
        "weight_decay": 1e-5,
        "fine_tune_lr": 1e-6,
        "model_folder": "saved/6/",
        "dataset_path": "datasets\wmt14_translate_de-en_train.csv"
    }

def get_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=True)
    num_added_tokens = tokenizer.add_tokens(['[START]', '[END]'], special_tokens=True)
    # num_added_tokens += tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    vocab_size = tokenizer.vocab_size + num_added_tokens
    return tokenizer, vocab_size

def get_causal_mask(shape, device):
    causal_mask = torch.triu(torch.ones((shape), dtype=torch.float), diagonal=1).to(device)
    causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf')).unsqueeze(0).unsqueeze(0)
    return causal_mask

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
    tokenized_sequence = tokenizer(sequence, max_length = 512, truncation = True)['input_ids']
    return torch.tensor(tokenized_sequence).to(device)

def visualize_attention_matrix(attention_matrix, sequence, tokenizer):
    sequence = sequence.squeeze(0)
    attention_matrix = np.mean(np.array(attention_matrix.cpu().detach().numpy()), axis=(0, 1)

    fig, ax = plt.subplots()
    im = ax.imshow(attention_matrix, cmap='viridis')
    ax.set_xticks(np.arange(len(sequence)))
    ax.set_yticks(np.arange(len(sequence)))
    ax.set_xticklabels([tokenizer.decode([token.item()]) for token in sequence], rotation=45)
    ax.set_yticklabels([tokenizer.decode([token.item()]) for token in sequence])
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_title('attention scores')
    plt.show()

def load_model_from_checkpoint(model_path, losses_path, model):
    if os.path.exists(model_path) and os.path.exists(losses_path):
        model.load_state_dict(torch.load(model_path))
        losses = pd.read_csv(losses_path)
        step = losses.shape[0]
        losses = losses['losses'].tolist()
        print("Loading model from checkpoint at step", step)
    elif os.path.exists(model_path) == False and os.path.exists(losses_path) == False:
        step = 0
        losses = []
        print("NOT loading model from checkpoint, creating new model at step", step)
    else:
        print("ERROR: one of either model and losses exists, both need to exist or not exist")
        return -1
    return model, step, losses

def entropy_loss(attention_matrix, entropy_weight):
    attention_entropy = -torch.sum(attention_matrix * torch.log(attention_matrix + 1e-12), dim=-1)
    entropy_loss = entropy_weight * torch.mean(attention_entropy)
    return entropy_loss
