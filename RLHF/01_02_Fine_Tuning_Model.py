from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import pandas as pd

# 1.Tokenize a text dataset
dataset = load_dataset("argilla/tripadvisor-hotel-reviews")
df = pd.DataFrame(dataset['train'])

# Split manually from train set
train_test_split = dataset["train"].train_test_split(test_size=0.2, seed=42)

tokenizer = AutoTokenizer.from_pretrained("openai-gpt")

# Add padding with the pad token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
   

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

# Tokenize the dataset
tokenized_datasets = train_test_split.map(tokenize_function, batched=True)

# 2.Fine-tuning for review classification

# AutoModelForCausalLM simplifies loading and switching models
model = AutoModelForCausalLM.from_pretrained("openai-gpt")
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)  # Resize the model's token embeddings to match the tokenizer 

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    gradient_accumulation_steps=4
)

# Define the train and test datasets
training_dataset = tokenized_datasets["train"].shuffle(seed=42).select([i for i in list(range(0, 1000))])
testing_dataset = tokenized_datasets["test"].shuffle(seed=42).select([i for i in list(range(0, 100))])

# Initialize the trainer class
trainer = Trainer(
# Add arguments to the class
    model=model,
    args=training_args,
    train_dataset=training_dataset,
    eval_dataset=testing_dataset
)