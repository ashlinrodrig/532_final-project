%%capture
!pip install unsloth
# Also get the latest nightly Unsloth!
!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git

!pip install datasets
!pip install evaluate
!pip install GPUtil
!pip install peft

from unsloth import FastLanguageModel
import torch

# Parameters
max_seq_length = 2048
load_in_4bit = True

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",  # Model name
    max_seq_length=max_seq_length,
    dtype=None,  # Let the system auto-detect the dtype
    load_in_4bit=load_in_4bit,
)

# LoRA configuration for model fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
)

import time
import psutil
import torch
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from evaluate import load
from GPUtil import getGPUs
from sklearn.metrics import accuracy_score
from evaluate import load
import matplotlib.pyplot as plt

# Loading training data
df = pd.read_csv("training_data.csv")

# Renaming columns for clarity
df.rename(columns={"description": "input_text", "target": "output_text"}, inplace=True)

df["output_text"] = df["output_text"].astype(str)

# Split the data into train and test sets (80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
print("training and test data ready")

# Tokenize function
def tokenize_function(example):
    return tokenizer(
        example["input_text"],
        text_target=example["output_text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Apply tokenization to both train and test datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
print("training and test data tokenized")

# Set format for PyTorch tensors
tokenized_train_dataset.set_format("torch")
tokenized_test_dataset.set_format("torch")
print("training and test data formatted using tensors")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

import torch

# Function to ensure tensors are in float32
def convert_to_float32(tensor):
    if tensor.dtype != torch.float32:
        tensor = tensor.to(torch.float32)
    return tensor

# Example tensor initialization (replace with your actual tensor or data pointers)
e_ptrs = torch.randn(10, 10, dtype=torch.float32)  
c_ptrs = torch.randn(10, 10, dtype=torch.float32) 

# Explicitly cast e_ptrs and c_ptrs to float32
e_ptrs = convert_to_float32(e_ptrs)
c_ptrs = convert_to_float32(c_ptrs)

# Now ensure the tensors involved in operations are of the same dtype (float32)
accum = torch.zeros((10, 10), dtype=torch.float32)  # Initialize accumulator in float32

# Perform the dot product operation (make sure both tensors are in float32)
accum = torch.matmul(e_ptrs, c_ptrs.t())  

# Print the result
print(accum)

from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
import torch

# Instead of .float16(), use Unsloth's precision handling
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_train_dataset,
    dataset_text_field="output_text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        fp16_full_eval=True,
        logging_steps=1,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

# Unsloth's chat template training
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)

# Debugging: Print model parameter dtypes before training
print("Model Parameter Dtypes:")
for name, param in model.named_parameters():
    print(f"{name}: {param.dtype}")

# Train with explicit error handling
try:
    trainer_stats = trainer.train()
except Exception as e:
    print(f"Training Error: {e}")
    import traceback
    traceback.print_exc()
