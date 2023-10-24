from transformers import pipeline
from pprint import pprint
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
import torch
import pandas as pd

myner = pipeline('ner', model='my_ner_model',aggregation_strategy='simple')

results = myner('Significant differences in ASC, DHA, and GSH contents between treatments (n¼3) and between root sections (n¼3) of girdled trees were analysed with the statistics program SPSS 16.0 for windows (Chicago, IL, USA). Prior to the test of significance with the Turkey test, the normality and homogeneity of the data were tested. Normality of the data was tested with the KolmogorovSmirnov test that includes correction of significance after Lilliefors and Shapiro-Wilk. Homogeneity of variance was tested with the Levene test. If homogeneity was not given, values were transferred using the natural logarithm. If homogeneity was still not given, the Games-Howell test was applied. Significant differences at P <0.05 are indicated.')



tokenizer = AutoTokenizer.from_pretrained('my_ner_model')
model = AutoModelForTokenClassification.from_pretrained('my_ner_model')

inputs = tokenizer('Significant differences in ASC, DHA, and GSH contents between treatments (n¼3) and between root sections (n¼3) of girdled trees were analysed with the statistics program SPSS 16.0 for windows (Chicago, IL, USA). Prior to the test of significance with the Turkey test, the normality and homogeneity of the data were tested. Normality of the data was tested with the KolmogorovSmirnov test that includes correction of significance after Lilliefors and Shapiro-Wilk. Homogeneity of variance was tested with the Levene test. If homogeneity was not given, values were transferred using the natural logarithm. If homogeneity was still not given, the Games-Howell test was applied. Significant differences at P <0.05 are indicated.', return_tensors="pt")
logits = model(**inputs).logits
logits
#print(inputs["input_ids"])
ids = inputs["input_ids"]
predictions = torch.argmax(logits, dim=2)
predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
#print(logits)
print(predicted_token_class)

df = pd.read_csv("softcite/dev-set/softcite-ner/annotations.tsv",delimiter="\t")
corpus_entities = df[df["filename"]=="PMC2826650"]
dev_entities = pd.DataFrame([], columns=["label","span"])
i=0
start_label = False
label = ""
word = ""
for prediction in predicted_token_class:
    if prediction!='O':
        start_label = True
        if prediction!=label:
            label = prediction
            word = word + tokenizer.decode(ids[0][i])
            dev_entities.append([prediction, word])
            word=""
        else:
            word = word + tokenizer.decode(ids[0][i])    
        #print(prediction+":"+tokenizer.decode(ids[0][i]))
    
    i=i+1