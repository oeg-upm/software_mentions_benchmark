import os
import random
import pandas as pd

# Specify the directory where your files are located
corpus="softcite"
directory = "datasets/corpus_research_software/"+corpus+"/"

# Get a list of all files in the directory
#file_list = os.listdir("directory"+"text-files/")
file_list = os.listdir("corpus/pwc/test-set/text-files/")

# Shuffle the list of files randomly
random.shuffle(file_list)

# Define the proportions for the two collections (e.g., 70% and 30%)
proportion_collection1 = 0.7
proportion_collection2 = 0.3

# Calculate the number of files for each collection
total_files = len(file_list)
num_files_collection1 = int(total_files * proportion_collection1)
num_files_collection2 = total_files - num_files_collection1

# Split the shuffled list into two collections
train_collection = file_list[:num_files_collection1]
test_collection = file_list[num_files_collection1:]

train_collection = [os.path.splitext(file)[0] for file in train_collection]
test_collection = [os.path.splitext(file)[0] for file in test_collection]

print(train_collection)
print(test_collection)

#df = pd.read_csv(directory+"annotations.tsv",delimiter="\t")
df = pd.read_csv("corpus/pwc/test-set/pwc-ner/annotations.tsv",delimiter="\t")

#train_df = df.applymap(lambda x: x if x["filename"] in train_collection)
train_df = df[df['filename'].isin(train_collection)]
test_df = df[df['filename'].isin(test_collection)]

#train_df.to_csv(directory+"train-set/"+corpus+"-ner/annotations.tsv", sep="\t", index=False)
#test_df.to_csv(directory+"test-set/"+corpus+"-ner/annotations.tsv", sep="\t", index=False)
train_df.to_csv("corpus/papers_with_code/train-set/ner/annotations.tsv", sep="\t", index=False)
test_df.to_csv("corpus/papers_with_code/test-set/ner/annotations.tsv", sep="\t", index=False)

