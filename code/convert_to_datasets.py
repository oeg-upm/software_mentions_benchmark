import pandas as pd
import os 
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
from datasets import Dataset
import ast

def generate_huggingface_dataset(dir,destination):
    print(directory)
    files = os.listdir(directory)

    dfs = []

    #df = pd.DataFrame()

    for file in files:
        print(file)
        dfs.append(pd.read_csv(directory+file))

        #pmc_data = pd.read_csv(directory+file)
        #df = df.append(pmc_data)
        #pd.concat([df,pmc_data])

    df =  pd.concat(dfs, ignore_index=True)

    df["labels"] = df["labels"].apply(lambda x: getLabel(x))

    columns = ["tokens"]

    for i in range(len(columns)):
        df[columns[i]] = df[columns[i]].apply(ast.literal_eval)

    print(df)
      
    hg = Dataset(pa.Table.from_pandas(df))


    hg.rename_column("labels","ner_tags")
    #hg.rename_column("__index_level_0__","id")

    
    hg.save_to_disk(destination)

def getLabel(label):
    new_label = []
    tokens = label.split(",")
    for token in tokens:
        token = token[token.index("'")+1:token.rindex("'")]
        new_label.append(label2id[token])
    return new_label


label2id = {'CLS': 0,
 'SEP': 1,
 'B-Application_Mention': 2,
 'I-Application_Mention': 3,
 'O': 4}


directory = "corpus/benchmark_v2/train-set/processed_data/"

generate_huggingface_dataset(directory,"corpus/benchmark_v2/benchmark_v2_train.hf")

directory = "corpus/benchmark_v2/test-set/processed_data/"

generate_huggingface_dataset(directory,"corpus/benchmark_v2/benchmark_v2_test.hf")



