from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

dataset = load_dataset("yelp_review_full")
print(dataset)


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):

    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))

small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)