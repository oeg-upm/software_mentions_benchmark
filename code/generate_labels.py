import csv

labels = {'CLS':0, 'SEP':1, 'O':2}
index = 3

with open("datasets/benchmark/train-set/benchmark-ner/annotations.tsv","r",newline='') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t', lineterminator='\n')

    for line in reader:
        if "B-"+line[2] not in labels or "I-"+line[2] not in labels:
            labels["B-"+line[2]]=index
            index = index + 1
            labels["I-"+line[2]]=index
            index = index + 1
    print(labels)
            

    