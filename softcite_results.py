import os
import glob
import requests

for path in glob.glob("SoMeSci/PLoS_sentences/*.txt"):
    print("Analyzing file "+str(path)+"\n")

    f = open(path,"r")
    content = f.read()
    f.close()

    content = content.replace("\"","'")

    softcite_path = "results/softcite/"
    results_file = path.split("/")[-1]
    results_file = results_file.replace(".txt",".json")

    #url = "curl -X POST -d \"text="+content+"\" -o "+softcite_path+results_file+" https://thesis.esteban.linkeddata.es/service/processSoftwareText"
    #print(url)
    url2 = "https://thesis.esteban.linkeddata.es/service/processSoftwareText"
    obj = {"text":content}
    results = requests.post(url2, data=obj)

    f = open(str(softcite_path)+str(results_file),"w")
    f.write(results.text)
    f.close()

    #os.system(url)