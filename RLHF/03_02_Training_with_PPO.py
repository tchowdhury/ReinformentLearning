from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig
import torch

# 1. Load model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load dataset
dataset = load_dataset("imdb", split="train[:1%]")

# 3. Reward function
def simple_reward_fn(texts):
    return [len(t)/100 for t in texts]

# 4. Configure PPO using PPOConfig
ppo_config = PPOConfig(
    learning_rate=1.41e-5,
    batch_size=4,
    mini_batch_size=2,
    
    #forward_batch_size=4,  # Add the forward batch size if needed
    # Optionally, you can set other PPO-specific parameters here.
)

# 5. Initialize PPOTrainer with config
ppo_trainer = PPOTrainer(
    model=model,
    ref_model=None,  # Optional: set if you're using a reference model for KL loss
    #learning_rate=ppo_config.learning_rate,
    batch_size=ppo_config.batch_size,
    mini_batch_size=ppo_config.mini_batch_size,
)

# 6. Tokenize dataset
tokenized_inputs = tokenizer(
    dataset["text"], padding="max_length", truncation=True, max_length=256, return_tensors="pt"
)

# 7. Training loop
for epoch in range(2):
    print(f"Epoch {epoch+1}")
    for i in range(0, len(tokenized_inputs["input_ids"]), ppo_config.batch_size):
        input_ids = tokenized_inputs["input_ids"][i:i+ppo_config.batch_size].to(model.device)
        attention_mask = tokenized_inputs["attention_mask"][i:i+ppo_config.batch_size].to(model.device)

        response_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )

        responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        rewards = simple_reward_fn(responses)

        stats = ppo_trainer.step(
            queries=tokenizer.batch_decode(input_ids, skip_special_tokens=True),
            responses=responses,
            rewards=rewards
        )

        print(f"Batch {i // ppo_config.batch_size + 1}: reward avg = {sum(rewards)/len(rewards):.3f}")

print("Done!")

# Save the model
model.save_pretrained("./ppo_trained_model")
tokenizer.save_pretrained("./ppo_trained_model")

