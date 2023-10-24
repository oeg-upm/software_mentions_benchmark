import csv
import pandas as pd



directory = "datasets/corpus_application_mention/softcite/test-set/softcite-ner/"
file = "annotations_old.tsv"
new_file = "annotations.tsv"
mapping = {"Application_Usage":"Application_Mention", "ProgrammingEnvironment_Usage":"Application_Mention","OperatingSystem_Usage":"Application_Mention","Extension":"Application_Mention","Release":"Version","PlugIn_Usage":"Application_Mention","AlternativeName":"Application_Mention","Abbreviation":"Application_Mention","PlugIn_Deposition":"Application_Mention","ProgrammingEnvironment_Mention":"Application_Mention"}
remove_labels = ["Citation"]

tsv_read = pd.read_csv(directory+file, sep='\t')
print(tsv_read.shape)
tsv_read = tsv_read.loc[tsv_read["label"]!="Citation"]
tsv_read = tsv_read.loc[tsv_read["label"]!="Version"]
tsv_read = tsv_read.loc[tsv_read["label"]!="Developer"]
tsv_read = tsv_read.loc[tsv_read["label"]!="URL"]
print(tsv_read.shape)
tsv_read = tsv_read.applymap(lambda x: mapping[x] if x in mapping else x)
tsv_read.to_csv(directory+new_file, sep="\t", index=False)


'''
with open(directory+file,"r",newline='') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t', lineterminator='\n')

    for line in reader:

'''