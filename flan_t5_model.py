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

# Converting target to string since Flan-T5 generates text
df["output_text"] = df["output_text"].astype(str)

# Split the data into train and test sets (80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
print("training and test data ready")

# Load the tokenizer
model_name = "google/flan-t5-small" # comment this when using the saved sine-tuned model
# model_name = "./flan-t5-pitstop" # uncomment this to use saved fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_name)


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

# Load the model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir="./flan-t5-pitstop",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_strategy="epoch",
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    gradient_accumulation_steps=2, 
    push_to_hub=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer
)

# Starting the training process
print("Started training the model")
trainer.train()
print("Training finished")

# Save the fine-tuned model
model.save_pretrained("./flan-t5-pitstop")
tokenizer.save_pretrained("./flan-t5-pitstop")

# Full Test Dataset Evaluation
def evaluate_full_test_data():
    print("Generating the outputs for test data in order to get the accuracy of the model")
    predictions = []
    references = []

    # print("length of test input data - ", len(tokenized_test_dataset))

    for i in range(len(tokenized_test_dataset)):
        test_sample = tokenized_test_dataset[i]

        input_text = test_sample["input_text"]
        target_text = test_sample["output_text"]

        # Tokenize the input text
        test_input = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        
        model.eval()
        
        # Generate prediction
        output = model.generate(**test_input, max_length=10)
        prediction = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        
        # Store prediction and actual output
        predictions.append(prediction)
        references.append(target_text)
        # print(f"generated prediction for {i}th row")

    accuracy = accuracy_score(references, predictions)
    print(f"Test Set Accuracy: {accuracy * 100:.2f}%")
    rouge = load("rouge")
    results = rouge.compute(predictions=predictions, references=references)
    print("Rouge results: ", results)  

# Call the evaluation function for full test data
evaluate_full_test_data()

# Measuring Latency, Throughput, and Memory Utilization during Prediction
def measure_performance():
    # Example query from test dataset
    print("Evaluating a sample to get the latency, throughput and memory utilization during prediction.")
    test_sample = test_df["input_text"].iloc[0]
    print(f"Test Sample: {test_sample}")

    # Start timing
    start_time = time.time()

    # Track memory utilization
    memory_before = psutil.virtual_memory().used
    gpu_memory_before = getGPUs()[0].memoryUsed if torch.cuda.is_available() else None

    # Tokenize and generate prediction
    test_input = tokenizer(test_sample, return_tensors="pt", padding=True, truncation=True).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Generate prediction
    output = model.generate(**test_input, max_length=10)
    prediction = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # End timing
    end_time = time.time()

    # Measure memory usage after inference
    memory_after = psutil.virtual_memory().used
    gpu_memory_after = getGPUs()[0].memoryUsed if torch.cuda.is_available() else None
    
    # Latency and memory usage
    latency = end_time - start_time
    memory_usage = memory_after - memory_before
    gpu_memory_usage = gpu_memory_after - gpu_memory_before if gpu_memory_before is not None else None

    print(f"Prediction: {prediction}")
    print(f"Latency: {latency:.4f} seconds")
    print(f"Memory Usage: {memory_usage / (1024 ** 2):.2f} MB")
    if gpu_memory_usage is not None:
        print(f"GPU Memory Usage: {gpu_memory_usage / (1024 ** 2):.2f} MB")

    # Throughput: Measure how many predictions can be made in 1 second
    start_time = time.time()
    for _ in range(100):  # Run 100 predictions to measure throughput
        _ = model.generate(**test_input, max_length=10)
    end_time = time.time()
    throughput = 100 / (end_time - start_time)
    print(f"Throughput: {throughput:.2f} predictions/second")

# Call the function to measure performance
measure_performance()

def log_memory_during_inference_whole_data(test_dataset, device):
    memory_log = []
    gpu_memory_log = []
    timestamps = []
    start_time = time.time()

    for i, test_sample in enumerate(test_dataset):
        # Extract input text
        input_text = test_sample["input_text"]

        # Tokenize the input
        test_input = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)

        # Measure memory usage before generating predictions
        memory_used = psutil.virtual_memory().used / (1024 ** 2)  # Convert to MB
        gpu_memory_used = getGPUs()[0].memoryUsed / (1024 ** 2) if torch.cuda.is_available() else 0

        # Generate the prediction
        model.generate(**test_input, max_length=10)

        # Measure memory usage after generating predictions
        memory_log.append(memory_used)
        gpu_memory_log.append(gpu_memory_used)
        timestamps.append(time.time() - start_time)

        # Print progress every 50 samples
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(test_dataset)} samples...")

    # Return memory logs and timestamps
    return timestamps, memory_log, gpu_memory_log

# subset_test_dataset = tokenized_test_dataset.select(range(10)) # testing for some small dataset first
timestamps, memory_log, gpu_memory_log = log_memory_during_inference_whole_data(tokenized_test_dataset, device)

print("printing the test results here to save for future purposes - ")
print("Timestamps - ", timestamps)
print("memory_log - ", memory_log)
print("gpu memory log - ", gpu_memory_log)
# Plot memory usage over time
plt.figure(figsize=(12, 6))

plt.plot(timestamps, memory_log, label="CPU Memory Usage (MB)", color="blue")
if any(gpu_memory_log):  # Check if GPU memory log is non-zero
    plt.plot(timestamps, gpu_memory_log, label="GPU Memory Usage (MB)", color="orange")

# Add labels, title, and legend
plt.title("Memory Usage Over Time During Inference (Whole Test Dataset)")
plt.xlabel("Time (s)")
plt.ylabel("Memory Usage (MB)")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Calculate statistics
peak_cpu_memory = max(memory_log)
average_cpu_memory = sum(memory_log) / len(memory_log)

peak_gpu_memory = max(gpu_memory_log) if any(gpu_memory_log) else 0
average_gpu_memory = sum(gpu_memory_log) / len(gpu_memory_log) if any(gpu_memory_log) else 0

# Display statistics
print(f"Peak CPU Memory Usage: {peak_cpu_memory:.2f} MB")
print(f"Average CPU Memory Usage: {average_cpu_memory:.2f} MB")
if peak_gpu_memory > 0:
    print(f"Peak GPU Memory Usage: {peak_gpu_memory:.2f} MB")
    print(f"Average GPU Memory Usage: {average_gpu_memory:.2f} MB")