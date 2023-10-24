from transformers import pipeline
from pprint import pprint
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer




def tokenize_and_align_labels(examples):
    label_all_tokens = True
    tokenized_inputs = tokenizer(list(examples["tokens"]), truncation=True, is_split_into_words=True) # Tokenize the given input tokens

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == '0':
                label_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
        
    tokenized_inputs["labels"] = labels
    return tokenized_inputs



dataset = load_dataset("PlanTL-GOB-ES/CoNLL-NERC-es")

language_model = 'PlanTL-GOB-ES/roberta-base-bne' # BETO


tokenizer = AutoTokenizer.from_pretrained(language_model, add_prefix_space=True, truncation=True,  max_length=512)

labels_list = "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"
label_num_list= list(range(0,len(labels_list)))

label2id={}
id2label={}

for label,num in zip(labels_list,label_num_list):
    label2id[label]=num
    id2label[num]=label

print(id2label)

task = "ner"


#train_tokenized_datasets = dataset_train.map(tokenize_and_align_labels, batched=True)
train_dataset = dataset["train"]

print(train_dataset[20])

#train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=False)

#print(train_tokenized_datasets)

