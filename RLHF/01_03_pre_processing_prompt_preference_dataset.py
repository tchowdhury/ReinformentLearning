from datasets import load_dataset

# 1. Preparing the preference dataset

# Load the dataset
preference_data = load_dataset("trl-internal-testing/hh-rlhf-helpful-base-trl-style", split="train")

# Define a function to extract the prompt
def extract_prompt(text):
    # Extract the prompt as the first element in the list    
    prompt = text[0]["content"]
    return prompt

# Apply the function to the dataset 
preference_data_with_prompt = preference_data.map(
    lambda sample: {**sample, 'prompt': extract_prompt(sample['chosen'])}
)

sample = preference_data_with_prompt.select(range(1))
print(sample['prompt'])