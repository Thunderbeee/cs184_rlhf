import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

dataset = load_dataset("nvidia/HelpSteer2")

max_length = 512
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def filter_data(data, tokenizer, max_length):
    filtered_data = []
    for sample in tqdm(data, desc="Filtering data"):
        if len(tokenizer.encode(sample["prompt"], sample["response"], truncation=False)) <= max_length:
            filtered_data.append(sample)
    return filtered_data

filtered_train_data = filter_data(dataset["train"], tokenizer, max_length)

class RewardModelDataset(Dataset):
    def __init__(self, data, attribute="helpfulness"):
        self.data = data
        self.attribute = attribute

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample["prompt"], sample["response"], sample[self.attribute]

train_dataset = RewardModelDataset(filtered_train_data)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

reward_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=1)
optimizer = AdamW(reward_model.parameters(), lr=1e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reward_model.to(device)

def train_reward_model(model, tokenizer, train_loader, num_epochs=1):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for prompts, responses, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs = tokenizer(prompts, responses, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device).unsqueeze(1).float()

            outputs = model(**inputs)
            loss = F.mse_loss(outputs.logits, labels)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}: Loss: {total_loss / len(train_loader)}")

train_reward_model(reward_model, tokenizer, train_loader, num_epochs=1)

reward_model.save_pretrained("./checkpoints/reward_model")
tokenizer.save_pretrained("./checkpoints/reward_model")
