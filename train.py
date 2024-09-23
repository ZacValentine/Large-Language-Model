import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from Model import Model
from config import get_config, get_tokenizer, get_preprocessed_sequence, get_causal_mask, get_tokenized_text, checkpoint, get_preprocessed_sequence_pd_csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

config =  get_config()
tokenizer, vocab_size = get_tokenizer()

model = Model(embed_dim=config["embed_dim"],
              num_blocks=config["num_blocks"], 
              num_heads=config["num_heads"], 
              ff_dim=config["ff_dim"], 
              # dropout_rate=conf["dropout_rate"],
              batch_size=config["batch_size"], 
              vocab_size=vocab_size).to(device)

# tokenized_text = get_tokenized_text(tokenizer, device)
# num_steps = int(tokenized_text.shape[0] / config["length"])
train_df = pd.read_csv('datasets/wmt14_translate_de-en_train.csv',lineterminator='\n')
num_steps = train_df.shape[0]
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["lr"])
losses = []
# best_loss_interval = 10 //<- how many regular intervals between this interval
model.train()

for epoch in range(config["num_epochs"]):
    for step in range(num_steps):
        optimizer.zero_grad()
        # sequence = get_preprocessed_sequence(tokenized_text, step, config["length"], device).unsqueeze(0) # will get an error when it reaches the end
        sequence = get_preprocessed_sequence_pd_csv(step, train_df, tokenizer, device).unsqueeze(0)
        causal_mask = get_causal_mask((sequence[:, :-1].shape[1], sequence[:, :-1].shape[1]), device)

        logits, probs = model(sequence[:, :-1], key_padding_mask = None, causal_mask = causal_mask)
        loss = criterion(logits.view(-1, vocab_size), sequence[:, 1:].view(-1))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % config["interval"] == 0 and step != 0:
            checkpoint(step, config["interval"], model, config["model_folder"], losses)
    print("epoch", epoch + 1, "completed")
