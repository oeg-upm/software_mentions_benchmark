from bs4 import BeautifulSoup
import re

tei_doc = "/Users/estebangonzalezguardia/projects/software_benchmark/softcite_corpus-full.tei.xml"
#CLEANR = re.compile('<.*?>') 

def write_brat_file(name, content, annotations):
    f = open("/Users/estebangonzalezguardia/projects/software_benchmark/somesci_files/"+str(name)+".ann","w")
    count = 1
    for annotation in annotations:
        id = "T"+str(count)
        type = ""
        if annotation.get("type") == "software":
            type = "Application_Mention"
        elif annotation.get("type") == "version":
            type = "Version"
        elif annotation.get("type") == "publisher":
            type = "Developer"
        elif annotation.get("type") == "url":
            type = "URL"
        start = content.index(annotation.get_text())
        end = start + len(annotation)
        text = annotation.get_text()

        f.write(str(id)+"    "+str(type)+" "+str(start)+" "+str(end)+" "+str(text)+"\n")
        count = count + 1
    f.close()

def write_txt_file(name, content):
    f = open("/Users/estebangonzalezguardia/projects/software_benchmark/somesci_files/"+str(name)+".txt","w")
    f.write(content)
    f.close

with open(tei_doc, 'r') as tei:
    soup = BeautifulSoup(tei, 'lxml')

    list_articles = soup.find_all(lambda tag: tag.get("type")=="article" and tag.get("subtype")=="pmc")

    for article in list_articles:
        print(article.title.get_text())
        pmc = article.find(type='PMC')
        text = article.find('text')
        print("Analizing text....")
        article_text = article.find('text')
        annotations = article_text.find_all(lambda tag: tag.name=="rs" and (tag.get("type")=="software" or tag.get("type")=="version" or tag.get("type")=="publisher" or tag.get("type")=="url"))
        print("Creating txt file ...")
        write_txt_file(pmc.get_text(),text.get_text())
        print("Creating BRAT file ....")
        write_brat_file(pmc.get_text(), text.get_text(), annotations)

        if len(annotations) == 0:
            print("No annotations detected:")

        print("**************")
        #break