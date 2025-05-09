from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# 1. Initializing the reward

# Load the GPT-1 model, "openai-gpt", for the sequence classification task using Hugging Face's AutoModelForSequenceClassification.
# Initialize the reward configuration using "output_dir" as the output directory, and set the token maximum length to 60.

model = AutoModelForSequenceClassification.from_pretrained("openai-gpt")
tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
config = RewardConfig(max_length=60, output_dir="./output_dir")

# 2. Loading and Tokenizing Dataset

# Add padding with the pad token
#tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Define a function to extract the prompt
def extract_prompt(text):
    # Extract the prompt as the first element in the list    
    prompt = text[0]["content"]
    return prompt

train_dataset = load_dataset("trl-internal-testing/hh-rlhf-helpful-base-trl-style", split="train")
train_data_with_prompt = train_dataset.map(
    lambda sample: {**sample, 'prompt': extract_prompt(sample['chosen'])}
)

eval_dataset = load_dataset("trl-internal-testing/hh-rlhf-helpful-base-trl-style", split="test")
eval_data_with_prompt = eval_dataset.map(
    lambda sample: {**sample, 'prompt': extract_prompt(sample['chosen'])}
)

def tokenize_function(examples):
    # Tokenize text field of the dataset
    return tokenizer(examples["chosen"], truncation=True, padding="max_length")

train_dataset = train_data_with_prompt.map(tokenize_function, batched=True)
eval_dataset = eval_data_with_prompt.map(tokenize_function, batched=True)


# 3. Setting up the reward trainer

reward_trainer = RewardTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=config,
)