import pandas as pd
import os
labels = ['Application_Mention','Developer','Version','URL']

#df = pd.read_csv("datasets/softcite/dev-set/softcite-ner/annotations.tsv",delimiter="\t")

#TRAIN DATASETS
print("**************TRAIN DATASETS**************")
df = pd.read_csv("datasets/softcite/train-set/softcite-ner/annotations.tsv",delimiter="\t")
files = os.listdir("datasets/softcite/train-set/text-files/")
files = [file.replace(".txt","") for file in files]
train_dataset = df[df["filename"].isin(files)]

print("Application-Mention:"+str(train_dataset[train_dataset["label"]=="Application_Mention"].size)+" entities.")
print("Developer:"+str(train_dataset[train_dataset["label"]=="Developer"].size)+" entities")
print("Version:"+str(train_dataset[train_dataset["label"]=="Version"].size)+" entities")
print("URL:"+str(train_dataset[train_dataset["label"]=="URL"].size)+" entities")
print("Total:"+str(train_dataset.size))

#TEST DATASETS
print("**************TEST DATASETS**************")
df = pd.read_csv("datasets/softcite/test-set/softcite-ner/annotations.tsv",delimiter="\t")
files = os.listdir("datasets/softcite/test-set/text-files/")
files = [file.replace(".txt","") for file in files]
test_dataset = df[df["filename"].isin(files)]

print("Application-Mention:"+str(test_dataset[test_dataset["label"]=="Application_Mention"].size)+" entities")
print("Developer:"+str(test_dataset[test_dataset["label"]=="Developer"].size)+" entities")
print("Version:"+str(test_dataset[test_dataset["label"]=="Version"].size)+" entities")
print("URL:"+str(test_dataset[test_dataset["label"]=="URL"].size)+" entities")
print("Total:"+str(test_dataset.size))

#DEV DATASETS
print("**************DEV DATASETS**************")
df = pd.read_csv("datasets/softcite/dev-set/softcite-ner/annotations.tsv",delimiter="\t")
files = os.listdir("datasets/softcite/dev-set/text-files/")
files = [file.replace(".txt","") for file in files]
#for file in files:
#    file_annotations = file.replace(".txt","")
dev_dataset = df[df["filename"].isin(files)]

print("Application-Mention:"+str(dev_dataset[dev_dataset["label"]=="Application_Mention"].size)+" entities")
print("Developer:"+str(dev_dataset[dev_dataset["label"]=="Developer"].size)+" entities")
print("Version:"+str(dev_dataset[dev_dataset["label"]=="Version"].size)+" entities")
print("URL:"+str(dev_dataset[dev_dataset["label"]=="URL"].size)+" entities")
print("Total:"+str(dev_dataset.size))