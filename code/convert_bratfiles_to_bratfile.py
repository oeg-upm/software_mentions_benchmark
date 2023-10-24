from os import listdir
from os.path import isfile, join
from tokenize import untokenize
import csv

#brat_files_directory = "datasets/benchmark/train-set/brat-files/"
#brat_file_directory = "datasets/benchmark/train-set/benchmark-ner/"

#brat_files_directory = "datasets/benchmark/dev-set/brat-files/"
#brat_file_directory = "datasets/benchmark/dev-set/benchmark-ner/"

brat_files_directory = "corpus/softcite/brat/"
brat_file_directory = "corpus/softcite/"

onlyfiles = [f for f in listdir(brat_files_directory) if isfile(join(brat_files_directory, f))]

with open(brat_file_directory+"annotations.tsv","w",newline='') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
    writer.writerow(["filename","mark","label","off0","off1","span"])
    for f in onlyfiles:
        fopen = open(brat_files_directory+str(f), "r")
        for line in fopen.readlines():
            #print(line)
            line = line.replace('\n','')
            line = line.replace('\t',' ')
            tokens = line.split(" ")
            selected_tokens = [token for token in tokens if len(token)>0]
            if len(selected_tokens) > 3:
                application_name = ''
                for token in selected_tokens[4::]:
                    application_name = application_name + token + " "
                application_name = application_name[:-1]
                
                if (selected_tokens[0].startswith("T")):
                    writer.writerow([f.replace('.ann',''),selected_tokens[0],selected_tokens[1],selected_tokens[2], selected_tokens[3], application_name])