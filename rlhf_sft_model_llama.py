import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AdamW,
    get_scheduler
)
from tqdm import tqdm
import matplotlib.pyplot as plt

MAX_LENGTH = 512
BATCH_SIZE = 6
LEARNING_RATE = 1e-5
NUM_EPOCHS = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = load_dataset("nvidia/HelpSteer2")

def filter_data(data, tokenizer, max_length):
    filtered_data = []
    for sample in tqdm(data, desc="Filtering data"):
        total_length = len(tokenizer.encode(sample["prompt"], sample["response"], truncation=False))
        if total_length <= max_length:
            filtered_data.append(sample)
    return filtered_data

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
filtered_train_data = filter_data(data["train"], tokenizer, MAX_LENGTH)
filtered_val_data = filter_data(data["validation"], tokenizer, MAX_LENGTH)

class RLHFDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample["prompt"], sample["response"]

rlhf_train_dataset = RLHFDataset(filtered_train_data)
rlhf_val_dataset = RLHFDataset(filtered_val_data)
rlhf_train_loader = DataLoader(rlhf_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
rlhf_val_loader = DataLoader(rlhf_val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
model.to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

num_training_steps = len(rlhf_train_loader) * NUM_EPOCHS
scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

reward_model_tokenizer = RobertaTokenizer.from_pretrained("./checkpoints/reward_model")
reward_model = RobertaForSequenceClassification.from_pretrained("./checkpoints/reward_model")
reward_model.eval()
reward_model.to(device)

def policy_gradient_loss_ppo(logits, generated, rewards, pad_token_id):
    logits[:, :, pad_token_id] = float("-inf")
    lprobs = torch.log_softmax(logits, dim=-1)
    generated_lprobs = lprobs.gather(-1, generated.unsqueeze(-1)).squeeze(-1)
    mask = generated != pad_token_id
    log_probs = (generated_lprobs * mask).sum(dim=-1)
    loss = -(log_probs * rewards).mean()
    return loss

def evaluate_model(model, reward_model, tokenizer, val_loader):
    model.eval()
    total_rewards = 0
    total_samples = 0

    with torch.no_grad():
        for prompts, responses in tqdm(val_loader, desc="Evaluating Validation Set"):
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(device)
            generated_tokens = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=MAX_LENGTH,
                do_sample=True,
                num_beams=1
            )

            generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            reward_inputs = reward_model_tokenizer(
                prompts,
                generated_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH
            ).to(device)

            rewards = reward_model(**reward_inputs).logits.view(-1)
            total_rewards += rewards.sum().item()
            total_samples += len(rewards)

    model.train()
    return total_rewards / total_samples

def train_rlhf_model(model, reward_model, tokenizer, train_loader, val_loader, num_epochs):
    val_rewards = []

    for epoch in range(num_epochs):
        model.train()
        for prompts, _ in tqdm(train_loader, desc=f"RLHF Epoch {epoch + 1}/{num_epochs}"):
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(device)
            generated_tokens = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=MAX_LENGTH,
                do_sample=True,
                num_beams=1
            )

            generated_tokens = F.pad(
                generated_tokens,
                (0, inputs["input_ids"].size(1) - generated_tokens.size(1)),
                value=tokenizer.pad_token_id
            )

            with torch.no_grad():
                generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                reward_inputs = reward_model_tokenizer(
                    prompts,
                    generated_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH
                ).to(device)

                rewards = reward_model(**reward_inputs).logits.view(-1)
                # Baseline
                rewards = rewards - rewards.mean()

            outputs = model(**inputs, labels=generated_tokens)
            loss = policy_gradient_loss(outputs.logits, generated_tokens, rewards, tokenizer.pad_token_id)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        val_performance = evaluate_model(model, reward_model, tokenizer, val_loader)
        val_rewards.append(val_performance)
        print(f"Epoch {epoch + 1} Validation Reward: {val_performance:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(val_rewards) + 1), val_rewards, marker='o', label='Validation Reward')
    plt.title("Validation Reward During RLHF Training")
    plt.xlabel("Epoch")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid()
    plt.savefig("./checkpoints/rlhf_reward_llama_2.png")
    plt.show()

train_rlhf_model(model, reward_model, tokenizer, rlhf_train_loader, rlhf_val_loader, NUM_EPOCHS)

rlhf_save_path = "./checkpoints/rlhf_model_llama_2"
os.makedirs(rlhf_save_path, exist_ok=True)
model.save_pretrained(rlhf_save_path)
tokenizer.save_pretrained(rlhf_save_path)
