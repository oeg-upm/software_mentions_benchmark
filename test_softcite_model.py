from fileinput import filename
import glob
import json
import csv

softcite_path = "results/softcite/"

def load_softcite_results(file):
    f = open(file,"r")
    content = f.read()
    f.close()
    return json.loads(content)

def load_somesci_results(file):
    dictionary = {}
    dictionary_without_positions = {}
    f = open(file,"r")
    for line in f.readlines():
        if line!="":
            tokens = line.split()
            if (tokens[0].startswith("T")):
                hashcode = str(tokens[4].lower())+"$-"+str(tokens[2])+"-"+str(tokens[3])
                dictionary[hashcode] = tokens[1]
                dictionary_without_positions[tokens[4].lower()] = tokens[1]
    return dictionary,dictionary_without_positions

f_results = open("summary_softcite.csv","w")
summary_info = ['file','true_positive','false_positive','fp with position','fp_type','fp_text','false_negative']
writer = csv.DictWriter(f_results, fieldnames=summary_info)
writer.writeheader()

results = {}

for path in glob.glob("SoMeSci/PLoS_sentences/*.ann"):

    print("Analyzing file "+str(path))
    
    results = {}

    #path = "SoMeSci/PLoS_sentences/PMC2841167.ann"

    tp = 0;fp=0;fp_position=0;fp_text=0;fp_type=0;fn=0
    
    somesci_results, somesci_results_no_position = load_somesci_results(path)

    filename = path.split("/")[-1]
    filename = filename.replace(".ann",".json")

    softcite_results_json = load_softcite_results(softcite_path+str(filename))

    for result in softcite_results_json["mentions"]:
        hashcode = str(result["software-name"]["rawForm"].lower())+"$-"+str(result["software-name"]["offsetStart"])+"-"+str(result["software-name"]["offsetEnd"])
        if hashcode in somesci_results:
            if (somesci_results[hashcode] == "SoftwareDependency" or somesci_results[hashcode] == "Abbreviation" or somesci_results[hashcode] == "ProgrammingLanguage" or somesci_results[hashcode] == "OperativeSystem" or somesci_results[hashcode] == "Application" or somesci_results[hashcode] =="AlternativeName"  or somesci_results[hashcode] == "Extension" or somesci_results[hashcode] == "Application_Usage"):
                tp = tp + 1
                del somesci_results[hashcode]
            else: 
                fp_type = fp_type + 1
                fp = fp + 1
        elif hashcode in somesci_results_no_position:
            fp_position = fp_position + 1
            fp = fp + 1
        else:
            fp = fp + 1
            fp_text = fp_text + 1

        try:
            hashcode = str(result["version"]["rawForm"])+"$-"+str(result["version"]["offsetStart"])+"-"+str(result["version"]["offsetEnd"])
            if hashcode in somesci_results:
                if (somesci_results[hashcode] == "Version"):
                    tp = tp + 1
                    del somesci_results[hashcode]
                else: 
                    fp_type = fp_type + 1
                    fp = fp + 1
            elif hashcode in somesci_results_no_position:
                fp_position = fp_position + 1
                fp = fp + 1
            else:
                fp = fp + 1
                fp_text = fp_text + 1
        except:
            print("Exception")
    
    fn = len(somesci_results)

    results["file"] = path
    results["true_positive"] = tp
    results["false_positive"]= fp
    results["fp with position"]= fp_position
    results["fp_type"] = fp_type
    results["fp_text"] = fp_text
    results["false_negative"] = fn

    writer.writerow(results)

    #break

