import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    AdamW,
    get_scheduler
)
from tqdm import tqdm
import matplotlib.pyplot as plt

MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = load_dataset("nvidia/HelpSteer2")

def filter_data(data, tokenizer, max_length):
    filtered_data = []
    for sample in tqdm(data, desc="Filtering data"):
        total_length = len(tokenizer.encode(sample["prompt"], sample["response"], truncation=False))
        if total_length <= max_length:
            filtered_data.append(sample)
    return filtered_data

tokenizer = AutoTokenizer.from_pretrained("t5-small")
tokenizer.pad_token = tokenizer.eos_token
filtered_train_data = filter_data(data["train"], tokenizer, MAX_LENGTH)
filtered_val_data = filter_data(data["validation"], tokenizer, MAX_LENGTH)

class SFTDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample["prompt"], sample["response"]

train_dataset = SFTDataset(filtered_train_data)
val_dataset = SFTDataset(filtered_val_data)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = T5ForConditionalGeneration.from_pretrained("t5-small")
model.to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

num_training_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

def train_sft(model, optimizer, train_loader, val_loader, num_epochs):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for prompts, responses in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            inputs = tokenizer(prompts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
            targets = tokenizer(responses, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)

            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=targets["input_ids"])
            loss = outputs.loss
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1} Training Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for prompts, responses in tqdm(val_loader, desc="Validating"):
                inputs = tokenizer(prompts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
                targets = tokenizer(responses, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)

                outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=targets["input_ids"])
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", marker="o")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker="o")
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig("./checkpoints/sft_loss.png")
    plt.show()

train_sft(model, optimizer, train_loader, val_loader, NUM_EPOCHS)

sft_save_path = "./checkpoints/sft_t5_model"
os.makedirs(sft_save_path, exist_ok=True)
model.save_pretrained(sft_save_path)
tokenizer.save_pretrained(sft_save_path)
