from transformers import pipeline
from metrics import Metrics
import pandas as pd
import os

labels = ['Application_Mention','Developer','Version','URL']
indicators = ["true_positives","false_positives", "false_positives_position", "false_positives_type", "false_negatives","false_negatives_position","false_negatives_type"]

metricsObject = Metrics(labels, indicators)

path = "datasets/softcite/dev-set/text-files/"
myner = pipeline('ner', model='models/softcite_model_e5_train_filtered',aggregation_strategy='simple')

df = pd.read_csv("datasets/softcite/dev-set/softcite-ner/annotations.tsv",delimiter="\t")

print("Application-Mention:"+str(df[df["label"]=="Application_Mention"].size)+" entities")
print("Developer:"+str(df[df["label"]=="Developer"].size)+" entities")
print("Version:"+str(df[df["label"]=="Version"].size)+" entities")
print("URL:"+str(df[df["label"]=="URL"].size)+" entities")


files = os.listdir("datasets/softcite/dev-set/text-files/")


i=0

true_positives = 0
false_positives = 0
false_negatives = 0
fn_type = 0
fn_position = 0
fp_type = 0
fp_position = 0

acc_true_positives = 0
acc_false_positives = 0
acc_false_negatives = 0
acc_fn_type = 0
acc_fn_position = 0
acc_fp_type = 0
acc_fp_position = 0

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

def printGlobalStatistics():
    print("TP:"+str(acc_true_positives))
    print("FP:"+str(acc_false_positives))
    print("FN:"+str(acc_false_negatives))
    print("FN (type):"+str(acc_fn_type))
    print("FN (position):"+str(acc_fn_position))
    print("FP (type):"+str(acc_fp_type))
    print("FP (position):"+str(acc_fp_position))


def printMetrics():
    print("Recall:"+str(recall))

for file in files:
    initStatistics()
    with open(path+file,'r') as f:
        text = f.read()
        results = myner(text[:1024])
#        results = myner(text)
        print("*******************************************************************")
        print(file)
        print(text)
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
                    metricsObject.addValue2Indicator(entity[2],"true_positives")
                    print("Text of corpus detected in predictions")
                elif len(ent21) > 0:
                    false_negatives = false_negatives + 1
                    metricsObject.addValue2Indicator(entity[2],"false_negatives")
                    fn_position = fn_position +1
                    metricsObject.addValue2Indicator(entity[2],"false_negatives_position")
                    print("Text of corpus detected in a different position")
                elif len(ent2) > 0:
                    false_negatives = false_negatives + 1
                    metricsObject.addValue2Indicator(entity[2],"false_negatives")
                    fn_type = fn_type + 1
                    metricsObject.addValue2Indicator(entity[2],"false_negatives_type")
                    print("Text of corpus detected with a different label")
                else:
                    false_negatives = false_negatives + 1
                    metricsObject.addValue2Indicator(entity[2],"false_negatives")
                    print("Text of corpus not detected in predictions")
                    
        printStatistics()
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
                    metricsObject.addValue2Indicator(prediction["entity_group"],"false_positives")
                    fp_position = fp_position + 1
                    metricsObject.addValue2Indicator(prediction["entity_group"],"false_positives_position")
                    print("Text of prediction detected in a different position")
                elif len(entities_word) > 0:
                    false_positives = false_positives + 1
                    metricsObject.addValue2Indicator(prediction["entity_group"],"false_positives")
                    fp_type = fp_type + 1
                    metricsObject.addValue2Indicator(prediction["entity_group"],"false_positives_type")
                    print("Text of prediction detected with a different type")
                else:
                    false_positives = false_positives + 1
                    metricsObject.addValue2Indicator(prediction["entity_group"],"false_positives")
                    print("Text of prediction not detected in corpus")
        #Update global variables
        acc_true_positives = acc_true_positives+true_positives
        acc_false_positives = acc_false_positives + false_positives
        acc_false_negatives = acc_false_negatives + false_negatives
        acc_fn_type = acc_fn_type + fn_type
        acc_fn_position = acc_fn_position + fn_position
        acc_fp_type = acc_fp_type + fp_type
        acc_fp_position = acc_fp_position + fp_position
    
    metricsObject.addMetrics2File(file)
    metricsObject.accumulateMetrics()
    metricsObject.resetMetrics()
    
    i=i+1
    if i==20:
        break

printGlobalStatistics()

precision = (acc_true_positives/(acc_true_positives+acc_false_positives))
recall = (acc_true_positives/(acc_true_positives+acc_false_negatives))

print("Precision"+str(precision))
print("Recall: "+str(recall))

acc_false_negatives = acc_false_negatives - acc_fn_position
acc_true_positives = acc_true_positives + acc_fn_position
acc_false_positives = acc_false_positives - acc_fp_position

precision = (acc_true_positives/(acc_true_positives+acc_false_positives))
recall = (acc_true_positives/(acc_true_positives+acc_false_negatives))

print("Precision"+str(precision))
print("Recall without position problem: "+str(recall))

metricsObject.calculateMetrics()
metrics = metricsObject.getMetricsFramework()

print(metrics)

#printMetrics()
#results = myner('Significant differences in ASC, DHA, and GSH contents between treatments (n¼3) and between root sections (n¼3) of girdled trees were analysed with the statistics program SPSS 16.0 for windows (Chicago, IL, USA). Prior to the test of significance with the Turkey test, the normality and homogeneity of the data were tested. Normality of the data was tested with the KolmogorovSmirnov test that includes correction of significance after Lilliefors and Shapiro-Wilk. Homogeneity of variance was tested with the Levene test. If homogeneity was not given, values were transferred using the natural logarithm. If homogeneity was still not given, the Games-Howell test was applied. Significant differences at P <0.05 are indicated.')

#print (results)

