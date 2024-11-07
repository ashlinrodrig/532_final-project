import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
# # Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B")

# # Replace 'YOUR_HF_TOKEN' with your actual Hugging Face token
# os.environ["HF_TOKEN"] = "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"

# Load the descriptions (after they were saved to 'lap_descriptions.csv')
lap_descriptions = pd.read_csv('lap_descriptions.csv', header=None)
lap_descriptions = lap_descriptions[0].tolist()

# Initialize the model and tokenizer
model_name = "EleutherAI/gpt-j-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize the texts
inputs = tokenizer(lap_descriptions, return_tensors="pt", padding=True, truncation=True)

# Set model to evaluation mode and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs = {key: val.to(device) for key, val in inputs.items()}

# Generate responses
outputs = model.generate(inputs['input_ids'], max_length=100)

# Decode and display results
generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
for i, text in enumerate(generated_texts):
    print(f"Generated Insight for Entry {i+1}:\n{text}\n")
