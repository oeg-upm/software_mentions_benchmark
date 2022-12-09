from transformers import AutoTokenizer
from SoMeNLP.somenlp.NER.models.multi_bert import BERTMultiTaskOpt2
from nltk.tokenize import sent_tokenize

from transformers import pipeline

import json
import torch

# Setup the path of the model and the tokenizer
MODEL_PATH = "/Users/estebangonzalezguardia/projects/software_benchmark/models/Gold-Multi-Simple-SciBERT/12-11-2022"
TOKENIZER_PATH = "/Users/estebangonzalezguardia/projects/software_benchmark/models/scibert_scivocab_cased"

class TestBinModel():

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        self.model = BERTMultiTaskOpt2.from_pretrained(MODEL_PATH)
        self.messages = []
        self.fp_position_reverse = 0

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

    def get_entities_pipeline(self, excerpt):
        myner = pipeline('ner', model='/Users/estebangonzalezguardia/projects/software_benchmark/models/Gold-Multi-Simple-SciBERT/12-11-2022')
        print(myner)

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
        #print(text)
        for i, ls in enumerate(lis):
            #print(f"\nlabels_map {labels_map[i][0]}\n")
            name = ''
            entity = ''
            idx = 0
            aux_text = text.lower()
            for j, l in enumerate(ls):
                out = labels_map[i][1][str(l)]
                if out != 'O':
                    part = tokens[j]
                    # print(_idx,out,part)
                    if out[0] == 'B' and name == '':
                        entity == out[2::]
                    if out[0] == 'B' and name != '':
                    #if out[0] == 'B':
                        _idx = aux_text.find(name)
                        _end = _idx + len(name)
                        if _idx == -1:
                            print(
                                'Error en las etiquetas/prediccion, NO DEBERIA DE OCURRIR')
                            name = part
                            entity = out[2::]
                            # print(aux_text)
                            continue
                        idx += _idx
                        
                        end = idx+len(name)
                        preds.append({'name': name, 'type': entity,
                                    'start': idx, 'end': end})
                        entity = out[2::]
                        #print(entity, idx, end, name)
                        idx = end
                        name = ''
                        aux_text = aux_text[_end::]
                        # print(aux_text)
                    if tokens[j][0:2] == '##':
                        part = tokens[j][2::]
                    name += part
                    entity = out[2::]
                    # print(out,tokens[j])
                elif name!='':
                    _idx = aux_text.find(name)
                    _end = _idx + len(name)
                    
                    idx += _idx
                    end = idx+len(name)
                    
                    preds.append({'name': name, 'type': entity,
                                    'start': idx, 'end': end})
                    #print(entity, idx, end, name)
                    idx = end
                    name = ''
                    entity = ''
                    aux_text = aux_text[_end::]
            if entity != '':
                _idx = aux_text.find(name)
                if _idx == -1:
                    print('Error en las etiquetas/prediccion, NO DEBERIA DE OCURRIR')
                    continue
                idx += _idx
                #Borrar en caso de fallo
                end = idx+len(name)
                preds.append({'name': name, 'type': entity,
                            'start': idx, 'end': end})
            break
                #print(entity, idx, idx+len(name), name)
            #else:
                #print('No existia ninguna "B-" en el map.')

        preds.sort(key=lambda x: x['start'])
        preds = [k for n, k in enumerate(preds) if k not in preds[n+1:]]
        #print(preds)
        return preds

    def eval_text(self, gs_file, text_file):
        self.messages = []
        summary = []
        debug = 0
        #Read gold standard file.
        true_positive = 0
        false_positive = 0
        false_negative = 0
        fp_type=0
        fp_text=0
        fp_position=0

        total = 0
        hashcode = ""

        original_type = ""

        #Build dictionary for BRAT file
        gs_dictionary = {}
        gs_dictionary_withoutposition = {}
        f_brat = open(gs_file)
        for line in f_brat.readlines():
            if line!="":
                tokens = line.split()
                hashcode = str(tokens[4].lower())+"$-"+str(tokens[2])+"-"+str(tokens[3])
                gs_dictionary[hashcode] = tokens[1]
                gs_dictionary_withoutposition[tokens[4].lower()] = tokens[1]
                #gs_dictionary[tokens[4].lower()] = tokens[1]

        f_brat.close()
        if debug: print("Dictionary:"+str(gs_dictionary))

        f_text = open(text_file)
        text = f_text.read()
        sentences = sent_tokenize(text)

        #Variable to calculate the position of the word respects the text, not the sentence
        sentence_cursor = 0

        for sentence in sentences:
            if debug: print("********************************************************")
            if debug: print("Sentence:"+str(sentence))
            _entities = self.get_entities(sentence)
            if debug: print("Entities:"+str(_entities))
            if debug: print("--------------------------------------------------------")
            for entity in _entities:
                #print(entity)
                if debug: print("Entity:"+str(entity))
                if debug: print (entity["name"])
                hashcode = str(entity["name"].lower())+"$-"+str(sentence_cursor+entity["start"])+"-"+str(sentence_cursor+entity["end"])
                if debug: print("hashcode:"+str(hashcode))
                if debug: print("Dictionary2:"+str(gs_dictionary))
                original_type = entity["type"]
                if entity["type"]=="SoftwareDependency" or entity["type"]=="Abbreviation" or entity["type"]=="ProgrammingLanguage" or entity["type"]=="OperativeSystem" or entity["type"]=="Application" or entity["type"]=="AlternativeName" or entity["type"]=="Extension":
                    entity["type"] = "Application_Mention"
                if entity["type"]=="Organization":
                    entity["type"] = "Developer"
                if entity["type"]=="Release":
                    entity["type"] = "Version"
                if hashcode in gs_dictionary:
                    gs_entity = gs_dictionary[hashcode]
                    #if debug: print(gs_entity)
                    #if debug: print(str(entity["type"]) + " " + str(gs_entity))
                    if entity["type"] == gs_entity:
                        true_positive = true_positive + 1
                        self.messages.append("TP: Match of "+str(entity["name"])+" with type="+str(entity["type"]))
                        if debug: print ("TP: Match of "+str(entity["name"])+" with type="+str(entity["type"]))
                    else: 
                        false_positive = false_positive + 1
                        fp_type = fp_type + 1
                        self.messages.append("FP: NOT match of "+str(entity["name"])+" with type_somesci="+str(original_type)+" and type_softcite="+str(gs_entity))
                        if debug: print ("FP: NOT match of "+str(entity["name"])+" with type="+str(entity["type"])+" and type="+str(gs_entity))
                else:
                    if entity["type"] == "Application_Mention" or entity["type"] == "Version" or entity["type"] == "URL":
                        false_positive = false_positive + 1
                        if (int(entity["start"]) > int(entity["end"])):
                            self.fp_position_reverse = self.fp_position_reverse + 1
                        if debug: print ("FP: NOT match of "+str(entity["name"]))
                        if entity["name"] in gs_dictionary_withoutposition:
                            fp_position = fp_position + 1
                            self.messages.append( "FP: NOT match in position of "+str(entity["name"])+" with code="+str(hashcode)+" and dictionary:"+str(gs_dictionary))
                        else:
                            self.messages.append( "FP: NOT match in text of "+str(entity["name"])+" with code="+str(hashcode)+" and dictionary:"+str(gs_dictionary))
                            fp_text = fp_text + 1


            sentence_cursor = sentence_cursor + len(sentence) + 1

        f_text.close()

        total = len(gs_dictionary)
        false_negative = total - true_positive - false_positive
        
        results = {}

        results["file"]=text_file
        results["true_positive"]=true_positive
        results["false_positive"]=false_positive
        results["fp with position"]=fp_position
        results["fp_type"]=fp_type
        results["fp_text"]=fp_text
        results["false_negative"]=false_negative

        return results

    def eval_text_pybsd(self, gs_file, text_file):
        self.messages = []
        summary = []
        debug = 0
        #Read gold standard file.
        true_positive = 0
        false_positive = 0
        false_negative = 0
        fp_type=0
        fp_text=0
        fp_position=0

        total = 0
        hashcode = ""

        original_type = ""

        #Build dictionary for BRAT file
        gs_dictionary = {}
        f_brat = open(gs_file)
        for line in f_brat.readlines():
            if line!="":
                tokens = line.split()
                hashcode = str(tokens[4].lower())+"$-"+str(tokens[2])+"-"+str(tokens[3])
                gs_dictionary[hashcode] = tokens[1]
                #gs_dictionary[tokens[4].lower()] = tokens[1]

        f_brat.close()
        if debug: print("Dictionary:"+str(gs_dictionary))

        f_text = open(text_file)
        text = f_text.read()
        sentences = sent_tokenize(text)

        #Variable to calculate the position of the word respects the text, not the sentence
        sentence_cursor = 0

        for sentence in sentences:
            if debug: print("********************************************************")
            if debug: print("Sentence:"+str(sentence))
            _entities = self.get_entities(sentence)
            if debug: print("Entities:"+str(_entities))
            if debug: print("--------------------------------------------------------")
            for entity in _entities:
                #print(entity)
                if debug: print("Entity:"+str(entity))
                if debug: print (entity["name"])
                hashcode = str(entity["name"].lower())+"$-"+str(sentence_cursor+entity["start"])+"-"+str(sentence_cursor+entity["end"])
                if debug: print("hashcode:"+str(hashcode))
                if debug: print("Dictionary2:"+str(gs_dictionary))
                original_type = entity["type"]
                if entity["type"]=="SoftwareDependency" or entity["type"]=="Abbreviation" or entity["type"]=="ProgrammingLanguage" or entity["type"]=="OperativeSystem" or entity["type"]=="Application" or entity["type"]=="AlternativeName" or entity["type"]=="Extension":
                    entity["type"] = "Application_Mention"
                if hashcode in gs_dictionary:
                    gs_entity = gs_dictionary[hashcode]
                    #if debug: print(gs_entity)
                    #if debug: print(str(entity["type"]) + " " + str(gs_entity))
                    if entity["type"] == gs_entity:
                        true_positive = true_positive + 1
                        self.messages.append("TP: Match of "+str(entity["name"])+" with type="+str(entity["type"]))
                        if debug: print ("TP: Match of "+str(entity["name"])+" with type="+str(entity["type"]))
                    else: 
                        false_positive = false_positive + 1
                        fp_type = fp_type + 1
                        self.messages.append("FP: NOT match of "+str(entity["name"])+" with type_somesci="+str(original_type)+" and type_softcite="+str(gs_entity))
                        if debug: print ("FP: NOT match of "+str(entity["name"])+" with type="+str(entity["type"])+" and type="+str(gs_entity))
                else:
                    if entity["type"] == "Application_Mention" or entity["type"] == "Version" or entity["type"] == "URL":
                        false_positive = false_positive + 1
                        self.messages.append( "FP: NOT match of "+str(entity["name"])+" with code="+str(hashcode)+" and dictionary:"+str(gs_dictionary))
                        if (int(entity["start"]) > int(entity["end"])):
                            self.fp_position_reverse = self.fp_position_reverse + 1
                        if debug: print ("FP: NOT match of "+str(entity["name"]))
                        if entity["name"] in hashcode:
                            fp_position = fp_position + 1
                        else:
                            fp_text = fp_text + 1


            sentence_cursor = sentence_cursor + len(sentence) + 1

        f_text.close()

        total = len(gs_dictionary)
        false_negative = total - true_positive
        
        results = {}

        results["file"]=text_file
        results["true_positive"]=true_positive
        results["false_positive"]=false_positive
        results["fp with position"]=fp_position
        results["fp_type"]=fp_type
        results["fp_text"]=fp_text
        results["false_negative"]=false_negative

        return results

    def view_messages(self):
        print("Reverse:"+str(self.fp_position_reverse))
        return self.messages






