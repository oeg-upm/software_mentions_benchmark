import csv
import pandas as pd



directory = "datasets/corpus_research_software/somesci/"
file = "annotations_old.tsv"
new_file = "annotations.tsv"
mapping = {"Application_Usage":"Application_Mention", "Extension":"Application_Mention","Release":"Version","AlternativeName":"Application_Mention","Abbreviation":"Application_Mention","ProgrammingEnvironment_Mention":"Application_Mention"}
remove_labels = ["Citation"]

tsv_read = pd.read_csv(directory+file, sep='\t')
tsv_read = tsv_read.applymap(lambda x: mapping[x] if x in mapping else x)
print(tsv_read.shape)
tsv_read = tsv_read.loc[tsv_read["label"]!="Citation"]
tsv_read = tsv_read.loc[tsv_read["label"]!="Version"]
tsv_read = tsv_read.loc[tsv_read["label"]!="Developer"]
tsv_read = tsv_read.loc[tsv_read["label"]!="URL"]
tsv_read = tsv_read.loc[tsv_read["label"]!="ProgrammingEnvironment_Usage"]
tsv_read = tsv_read.loc[tsv_read["label"]!="OperatingSystem_Usage"]
tsv_read = tsv_read.loc[tsv_read["label"]!="PlugIn_Usage"]
print(tsv_read.shape)
tsv_read.to_csv(directory+new_file, sep="\t", index=False)



'''
with open(directory+file,"r",newline='') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t', lineterminator='\n')

    for line in reader:

'''