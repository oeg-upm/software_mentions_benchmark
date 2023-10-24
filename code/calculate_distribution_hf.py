from datasets import load_from_disk

train_dataset = load_from_disk("datasets/softcite_train.hf")

labels= train_dataset["labels"]

total = 0
nLabels = 0

for label_set in labels:
    for label in label_set:
        total = total + 1
        if label != 10:
            nLabels = nLabels + 1

print("Labels:"+str(nLabels)+"("+str(nLabels/total)+"%)")
print("Total:"+str(total))