from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer,AutoModelForCausalLM


# Set the model name
model_name = 'lvwerra/gpt2-imdb-pos-v2' # the RLHF-pretrained model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a text generation pipeline
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

review_prompt = "Surprisingly, the film"

# Generate a continuation of the review
generated_text = text_generator(review_prompt, max_length=20, truncation=True)
print(generated_text)
print(f"Generated Review Continuation: {generated_text[0]['generated_text']}")

# Create a sentiment analysis pipeline
sentimental_model_name = 'lvwerra/distilbert-imdb' # the sentimental-pretrained model
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentimental_model_name)
sentiment_analyzer = pipeline('sentiment-analysis', model=sentiment_model, tokenizer=tokenizer)

# Classify the sentiment of the review
sentiment = sentiment_analyzer(generated_text[0]['generated_text'])
print(f"Sentiment Analysis Result: {sentiment}")