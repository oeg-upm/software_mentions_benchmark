from transformers import AutoTokenizer
from SoMeNLP.somenlp.NER.models.multi_bert import BERTMultiTaskOpt2
from nltk.tokenize import sent_tokenize

import json
import torch

# Setup the path of the model and the tokenizer
MODEL_PATH = "/Users/estebangonzalezguardia/projects/software_benchmark/models/Gold-Multi-Simple-SciBERT/12-11-2022"
TOKENIZER_PATH = "/Users/estebangonzalezguardia/projects/software_benchmark/models/scibert_scivocab_cased"

class TestBinModel():

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        self.model = BERTMultiTaskOpt2.from_pretrained(MODEL_PATH)

    def load_encoding(self):
        # Opening JSON file
        f = open(f'{MODEL_PATH}/encoding.json')
        # returns JSON object as
        # a dictionary
        data = json.load(f)
        f.close()
        # Iterating through the json
        # list
        labels_map = [("software", data['tag2name']['software']), ("soft_type", data['tag2name']
                                                                ['soft_type']), ("mention_type", data['tag2name']['mention_type'])]
        return labels_map

    def get_entities(self, excerpt):
        text = excerpt
        # genera los tokens sin el inicial y el final [cls] y [sep]
        tokens = self.tokenizer.tokenize(text)
        inputs = self.tokenizer(text, return_tensors="pt")
        out = self.model(**inputs)
        logits = out[1:-2]
        tag_seq = [torch.argmax(logits[i], axis=2) for i in range(len(logits))]
        lis = [tag_seq[i].tolist()[0] for i in range(len(tag_seq))]
        lis = [lis[i][1:-1] for i in range(len(lis))]

        labels_map = self.load_encoding()

        preds = []
        print(text)
        for i, ls in enumerate(lis):
            print(f"\nlabels_map {labels_map[i][0]}\n")
            name = ''
            entity = ''
            idx = 0
            aux_text = text.lower()
            for j, l in enumerate(ls):
                out = labels_map[i][1][str(l)]
                if out != 'O':
                    part = tokens[j]
                    # print(_idx,out,part)
                    if out[0] == 'B' and name != '':
                        _idx = aux_text.find(name)
                        _end = _idx + len(name)
                        if _idx == -1:
                            print(
                                'Error en las etiquetas/prediccion, NO DEBERIA DE OCURRIR')
                            name = part
                            # print(aux_text)
                            continue
                        idx += _idx
                        entity = out[2::]
                        end = idx+len(name)
                        preds.append({'name': name, 'type': entity,
                                    'start': idx, 'end': end})
                        print(entity, idx, end, name)
                        idx = end
                        name = ''
                        aux_text = aux_text[_end::]
                        # print(aux_text)
                    if tokens[j][0:2] == '##':
                        part = tokens[j][2::]
                    name += part
                    # print(out,tokens[j])
            if entity != '':
                _idx = aux_text.find(name)
                if _idx == -1:
                    print('Error en las etiquetas/prediccion, NO DEBERIA DE OCURRIR')
                    continue
                idx += _idx
                preds.append({'name': name, 'type': entity,
                            'start': idx, 'end': end})
                print(entity, idx, idx+len(name), name)
            else:
                print('No existia ninguna "B-" en el map.')

        preds.sort(key=lambda x: x['start'])
        preds = [k for n, k in enumerate(preds) if k not in preds[n+1:]]
        print(preds)
        return preds

    def eval_text(self, gs_file, text_file):
        summary = []
        debug = 1
        #Read gold standard file.
        true_positive = 0
        false_positive = 0
        false_negative = 0
        total = 0

        gs_dictionary = {}
        f_brat = open(gs_file)
        for line in f_brat.readlines():
            if line!="":
                tokens = line.split()
                gs_dictionary[tokens[4]] = tokens[1]

        f_brat.close()
        
        f_text = open(text_file)
        text = f_text.read()
        sentences = sent_tokenize(text)
        for sentence in sentences:
            if debug: print("********************************************************")
            if debug: print("Sentence:"+str(sentence))
            _entities = self.get_entities(sentence)
            if debug: print("Entities:"+str(_entities))
            if debug: print("--------------------------------------------------------")
            for entity in _entities:
                if debug: print (entity["name"])
                if entity["name"] in gs_dictionary:
                    gs_entity = gs_dictionary[entity["name"]]
                    if debug: print(gs_entity)
                    if debug: print(str(entity["type"]) + " " + str(gs_entity))
                    if (entity["type"] == gs_entity):
                        true_positive = true_positive + 1
                        if debug: print ("Match!")
                    else: 
                        false_positive = false_positive + 1
                else:
                    false_positive = false_positive + 1
            total = len(gs_dictionary)
            false_negative = total - true_positive
        f_text.close()
        print("SUMMARY")
        print("TP:"+str(true_positive))
        print("FP:"+str(false_positive))
        print("FN:"+str(false_negative))

        results = {}

        results["file"]=text_file
        results["true_positive"]=true_positive
        results["false_positive"]=false_positive
        results["false_negative"]=false_negative

        return results

        summary.append(results)
        print (summary)






