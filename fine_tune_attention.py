import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from Model import Model
from config import get_config, get_tokenizer, get_causal_mask, checkpoint, get_preprocessed_sequence_pd_csv, load_model_from_checkpoint, entropy_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

config =  get_config()
tokenizer, vocab_size = get_tokenizer()

model = Model(embed_dim=config["embed_dim"],
              num_blocks=config["num_blocks"], 
              num_heads=config["num_heads"], 
              ff_dim=config["ff_dim"], 
              dropout_rate=config["dropout_rate"],
              batch_size=config["batch_size"], 
              vocab_size=vocab_size).to(device)
model, start_step, losses = load_model_from_checkpoint(config["model_folder"] + "model.pth", config["model_folder"] + "losses.csv", model)

train_df = pd.read_csv(config["dataset_path"],lineterminator='\n')
num_steps = train_df.shape[0]
criterion = nn.CrossEntropyLoss()

for name, param in model.named_parameters():
    if 'attention' not in name: 
        param.requires_grad = False
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["fine_tune_lr"], weight_decay=config["weight_decay"])

model.train()

for epoch in range(config["num_epochs"]):
    for step in range(start_step, num_steps):
        optimizer.zero_grad()

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
