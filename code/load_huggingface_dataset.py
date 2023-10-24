from datasets import load_from_disk

train_dataset = load_from_disk("datasets/softcite_train.hf")

print(train_dataset["labels"])