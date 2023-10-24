from transformers import pipeline
from pprint import pprint
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer
import ast
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from datasets import load_metric
import numpy as np




def tokenize_and_align_labels(examples):
    label_all_tokens = True
    tokenized_inputs = tokenizer(list(examples["tokens"]), truncation=True, is_split_into_words=True, max_length=512) # Tokenize the given input tokens

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
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


#dataset_train = load_from_disk("datasets/huggingface/somesci/somesci_train.hf")
#dataset_train = load_from_disk("datasets/huggingface/softcite/softcite_train.hf")
#dataset_train = load_from_disk("datasets/corpus_research_software/benchmark/benchmark_train.hf")
dataset_train = load_from_disk("datasets/corpus_research_software/benchmark/benchmark_train.hf")

dataset_train = dataset_train.rename_column("labels","ner_tags")

#dataset_train = dataset_train.rename_column("__index_level_0__","id")


columns = ["tokens","ner_tags"]

language_model = 'allenai/scibert_scivocab_uncased' # BETO
#language_model = 'allenai/cs_roberta_base'


tokenizer = AutoTokenizer.from_pretrained(language_model, add_prefix_space=True, truncation=True,  max_length=512)

#labels_list = "CLS","SEP","B-Application_Mention","I-Application_Mention","B-Developer","I-Developer","B-URL","I-URL","B-Version","I-Version","O"
labels_list = "CLS","SEP","B-Application_Mention","I-Application_Mention","O"
label_num_list= list(range(0,len(labels_list)))

label2id={}
id2label={}

for label,num in zip(labels_list,label_num_list):
    label2id[label]=num
    id2label[num]=label


task = "ner"
#dataset_train["tokens"][0][0]="16.0"
#dataset_train["ner_tags"][0][0]="B-Version"


train_tokenized_datasets = dataset_train.map(tokenize_and_align_labels, batched=True)

#dataset_test = load_from_disk("somesci_test.hf")
#dataset_test = load_from_disk("datasets/huggingface/softcite/softcite_test.hf")
#dataset_test = load_from_disk("corpus/softcite/softcite_economics_test.hf")
dataset_test = load_from_disk("datasets/corpus_research_software/benchmark/benchmark_test.hf")
dataset_test = dataset_test.rename_column("labels","ner_tags")
#dataset_test = dataset_test.rename_column("__index_level_0__","id")

test_tokenized_datasets = dataset_test.map(tokenize_and_align_labels, batched=True)

#dataset_valid = load_from_disk("benchmark_dev.hf")
#dataset_valid = load_from_disk("datasets/huggingface/benchmark/benchmark_dev.hf")

#dataset_valid = load_from_disk("datasets/huggingface/softcite/softcite_dev.hf")

#dataset_valid = dataset_valid.rename_column("labels","ner_tags")
#dataset_valid = dataset_valid.rename_column("__index_level_0__","id")

#valid_tokenized_datasets = dataset_valid.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained(language_model, num_labels=len(labels_list), id2label = id2label, label2id = label2id)

batch_size=16 
epochs= 3

args = TrainingArguments(
    "spanish-ner",
    evaluation_strategy = "epoch",
    save_strategy="no",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=1e-5,
    learning_rate=1e-4,
    debug="underflow_overflow" )

data_collator = DataCollatorForTokenClassification(tokenizer)

metric = load_metric("seqeval")
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [[labels_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[labels_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}

trainer = Trainer(
    model,
    args,
    train_dataset=train_tokenized_datasets,
    eval_dataset=test_tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics   
)

trainer.train()

trainer.save_model('benchmark_v1_train-benchmark_v1_test-cs_scibert_base')