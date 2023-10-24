from os import listdir
from os.path import isfile, join
import csv

onlyfiles = [f for f in listdir("softcite/") if isfile(join("softcite/", f))]

with open("softcite/annotations.tsv","w",newline='') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
    for f in onlyfiles:
        fopen = open("softcite/"+str(f), "r")
        for line in fopen.readlines():
            tokens = line.split(" ")
            #writer.writerow([f,tokens[0],tokens[1],tokens[2], tokens[3], tokens[4::]])