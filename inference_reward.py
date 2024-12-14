from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

reward_model_path = "./checkpoints/reward_model"
reward_model_tokenizer = RobertaTokenizer.from_pretrained(reward_model_path)
reward_model = RobertaForSequenceClassification.from_pretrained(reward_model_path)
reward_model.eval()
reward_model.to(device)

outputs = [
    # "A place of history and culture, of art and gastronomy, of love and passion. A land of mountains, valleys and plains. A land of wine and cheese, of gastronomy and gastronomy, of art and culture. A land of history and tradition, of beauty and charm. A land where you can enjoy the best of both worlds. A land of contrasts, of contrasts that make you fall in love. A land where you can enjoy the best of",
    # "I am a student at the University of Bordeaux, France and I am interested in a career in the field of environmental engineering. I am very interested in your work and I would love to get to know you and learn more about your research. I am a graduate student at the University of Bordeaux, France. I am interested in the field of environmental engineering and I am currently working on my master’s thesis. I am very interested in your work and I would love to",
    "C\#",
    "C\# is a type of code that is used to generate data from the data."
]

context = "Please rate the following outputs based on their helpfulness."

inputs = reward_model_tokenizer(
    [context] * len(outputs),
    outputs,               
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
).to(device)

with torch.no_grad():
    scores = reward_model(**inputs).logits.squeeze(-1)

for i, score in enumerate(scores):
    print(f"Output {i}: Reward Score = {score.item():.4f}")
