# model_training.py

import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

def initialize_model(num_labels=5, learning_rate=1e-5, epsilon=1e-8):
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False,
    )
    # 检查是否有MPS（Apple Silicon）或CUDA可用，否则使用CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model.to(device)

    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,
                      eps=epsilon)
    return model, optimizer

def train_model(model, train_dataset, val_dataset, optimizer,device, epochs=4, batch_size=16):
    train_dataloader = DataLoader(
                train_dataset,  
                sampler = RandomSampler(train_dataset),
                batch_size = batch_size 
            )

    validation_dataloader = DataLoader(
                val_dataset, 
                sampler = RandomSampler(val_dataset), 
                batch_size = batch_size 
            )

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=0, 
                                                num_training_steps=total_steps)

    
    for epoch_i in range(epochs):
        model.train()
        total_train_loss = 0

        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            # 使用 outputs 对象获取 loss 和 logits
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)

            loss = outputs.loss
            logits = outputs.logits

            total_train_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        print(f"Epoch {epoch_i + 1} completed. Average training loss: {total_train_loss / len(train_dataloader)}")

    return model

def save_model(model, tokenizer, output_dir):
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
