from datasets import Dataset,load_from_disk

train_dataset = load_from_disk("datasets/softcite_train.hf")


i = 0
threshold = 50
labels = train_dataset["labels"]
nLabels = 0
exclude_indexes = []

print (len(train_dataset))

for label_set in labels:
    for label in label_set:
        if label != 10:
            nLabels = nLabels + 1
    if nLabels == 0 and i > 50:
        exclude_indexes.append(i)
    nLabels = 0
    i = i + 1

train_dataset_new = train_dataset.select(
    (
        i for i in range(len(train_dataset)) 
        if i not in set(exclude_indexes)
    )
)

print (len(train_dataset_new))
train_dataset_new.save_to_disk("datasets/softcite_train_new.hf")