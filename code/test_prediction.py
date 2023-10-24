from transformers import pipeline
from metrics import Metrics
import pandas as pd
import os
import pysbd


path = "softcite/dev-set/text-files/"
myner = pipeline('ner', model='my_ner_model',aggregation_strategy='simple')

df = pd.read_csv("softcite/dev-set/softcite-ner/annotations.tsv",delimiter="\t")



true_positives = 0
false_positives = 0
false_negatives = 0
fn_type = 0
fn_position = 0
fp_type = 0
fp_position = 0

recall = 0

def initStatistics():
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    fn_type = 0
    fn_position = 0
    fp_type = 0
    fp_position = 0

def printStatistics():
    print("TP:"+str(true_positives))
    print("FP:"+str(false_positives))
    print("FN:"+str(false_negatives))
    print("FN (type):"+str(fn_type))
    print("FN (position):"+str(fn_position))
    print("FP (type):"+str(fp_type))
    print("FP (position):"+str(fp_position))

file = "PMC4455856.txt"
initStatistics()
sequencer = pysbd.Segmenter(language="en", clean=False)

with open(path+file,'r') as f:
    text = f.read()
 #   results = myner(text[:512])
    results = []
    results = myner(text)

    print("*******************************************************************")
    print(text)
    
    sentences = sequencer.segment(text)

    '''
    for sentence in sentences:
        res = myner(sentence)
        if res != []:
            for r in res:
                results.append(r)
'''
    print(results)
            
    file_annotations = file.replace(".txt","")
    corpus_entities = df[df["filename"]==file_annotations]

    #Detection of true positives and false negatives
    if corpus_entities.size > 0:
        for entity in corpus_entities.values:
            print("Text:"+entity[5]+ " annotated as "+entity[2])
            ent2 = [x for x in results if x["word"] == entity[5].lower()]
            ent21 = [x for x in results if x["word"] == entity[5].lower() and  x["entity_group"] == entity[2]] 
            ent22 = [x for x in results if x["word"] == entity[5].lower() and  x["entity_group"] == entity[2] and x["start"]== entity[3]] 
            if len(ent22) > 0:
                true_positives = true_positives + 1
                print("Text of corpus detected in predictions")
            elif len(ent21) > 0:
                false_negatives = false_negatives + 1
                fn_position = fn_position +1
                print("Text of corpus detected in a different position")
            elif len(ent2) > 0:
                false_negatives = false_negatives + 1
                fn_type = fn_type + 1
                print("Text of corpus detected with a different label")
            else:
                false_negatives = false_negatives + 1
                print("Text of corpus not detected in predictions")
                    
    #Detection of true positives and false positives. Number of true positives must match
    print("--------------FALSE POSITIVES------------------")

    corpus_entities["span"]=corpus_entities["span"].apply(lambda x: x.lower())

    if len(results) > 0:
        for prediction in results:
            print("Text:"+prediction["word"]+ " annotated as "+prediction["entity_group"])
               
            entities_word = [x for x in corpus_entities.values if x[5] == prediction["word"]]
            entities_label = [x for x in corpus_entities.values if x[5] == prediction["word"] and  x[2] == prediction["entity_group"]] 
            entities_position = [x for x in corpus_entities.values if x[5] == prediction["word"] and  x[2] == prediction["entity_group"] and x[3]== prediction["start"]] 
            
            if len(entities_position) > 0:
                print("Text of prediction detected")
            elif len(entities_label) > 0:
                false_positives = false_positives + 1
                fp_position = fp_position + 1
                print("Text of prediction detected in a different position")
            elif len(entities_word) > 0:
                false_positives = false_positives + 1
                fp_type = fp_type + 1
                print("Text of prediction detected with a different type")
            else:
                false_positives = false_positives + 1
                print("Text of prediction not detected in corpus")
    
printStatistics()

