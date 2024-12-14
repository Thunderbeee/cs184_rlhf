# import torch
# from transformers import AutoTokenizer, T5ForConditionalGeneration

# pretrained_model_name = "t5-large"
# rlhf_model_path = "./checkpoints/rlhf_model"

# tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
# pretrained_model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name)
# rlhf_model = T5ForConditionalGeneration.from_pretrained(rlhf_model_path)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# pretrained_model.to(device)
# rlhf_model.to(device)

# prompts = [
#     "who are you?"
# ]

# def generate_responses(model, tokenizer, prompts, max_length=64):
#     responses = []
#     for prompt in prompts:
#         inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
#         outputs = model.generate(inputs["input_ids"], max_length=max_length, num_beams=8, early_stopping=True)
#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         responses.append(response)
#     return responses

# print("Generating responses from Pretrained T5...")
# pretrained_responses = generate_responses(pretrained_model, tokenizer, prompts)

# print("Generating responses from RLHF T5...")
# rlhf_responses = generate_responses(rlhf_model, tokenizer, prompts)

# print("\nComparison of T5 Outputs Before and After RLHF:")
# for i, prompt in enumerate(prompts):
#     print(f"\nPrompt: {prompt}")
#     print(f"Pretrained T5: {pretrained_responses[i]}")
#     print(f"RLHF T5: {rlhf_responses[i]}")


from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
rlhf_model = T5ForConditionalGeneration.from_pretrained("./checkpoints/sft_t5_model")

def answer_question(model_choice,question):
    input_text = f"question: {question}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output_ids = model_choice.generate(input_ids, max_length=128)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return answer

question = "introduce France"
# context = "France is a country in Europe. Its capital is Paris, which is known for its art, fashion, and culture."

answer = answer_question(model, question)
answer_rlhf = answer_question(rlhf_model, question)
print(f"Question: {question}")
print(f"Answer: {answer}")
print(f"Answer RLHF: {answer_rlhf}")

# from transformers import T5ForConditionalGeneration, T5Tokenizer

# model_name = 't5-large'
# tokenizer = T5Tokenizer.from_pretrained(model_name)
# model = T5ForConditionalGeneration.from_pretrained(model_name)
# input_text = "summarize: The T5 (Text-To-Text Transfer Transformer) model by Hugging Face is a versatile tool that frames various NLP tasks into a unified text-to-text format. This design allows you to apply T5 to tasks like translation, summarization, and question answering using a consistent approach. Here's how you can utilize T5 for different applications."
# inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
# outputs = model.generate(inputs, max_length=20, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True, temperature=0.5, do_sample=True)
# summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print("Summary:", summary)
