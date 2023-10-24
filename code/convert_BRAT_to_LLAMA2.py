import pandas
import json

directory = "datasets/corpus_research_software/benchmark/test-set/"
results=[]

def insert_text(text):
    annotation["input"]=text
    annotation["output"]=text

    #It works because positions are ordered

def insertAnnotation(start, end, span, shift):
    annotation["output"]=annotation["output"][:(start+shift)]+"@@"+annotation["output"][(start+shift):(end+shift)]+"##"+annotation["output"][(end+shift):]

def extract_string(filename):
    #open text file in read mode
    text_file = open(directory+"text-files/"+filename+".txt", "r", encoding="utf8")
    
    #read whole file to a string
    text = text_file.read()
    
    #close file
    text_file.close()
    

    return text

data=pandas.read_csv(directory+"benchmark-ner/annotations.tsv",sep='\t')

last_file = ""
text = ""
shift = 0

annotation={"instruction":""}

print(data)

for index,item in data.iterrows():
    if last_file != item["filename"]:
        if last_file != "":
            results.append(annotation)
            annotation={"instruction":""}
        shift = 0
        text = extract_string(item["filename"])
        insert_text(text)
        insertAnnotation(item["off0"], item["off1"], item["span"], shift)
        shift = shift+4
        print(item["filename"])
        last_file = item["filename"]
    else:
        insertAnnotation(item["off0"], item["off1"], item["span"], shift)
        shift = shift+4

print(results)

results_json = json.dumps(results)

with open('corpus_llm_benchmark_test', 'w',encoding="utf-8") as file:
    file.write(results_json)




    
