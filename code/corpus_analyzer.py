import pandas as pd

statistics = {}

def get_statistics(dataframe):
    response = {}
    filtered_df = dataframe[dataframe['label'] == "Application_Mention"]
    frequency = filtered_df['span'].value_counts()
    response["diversity"]=frequency.size
    response["#annotations"]=filtered_df.shape[0]
    response["mentions"]=frequency.to_dict()
    return response


df = pd.read_csv("datasets/somesci/train-set/somesci-ner/annotations.tsv",delimiter="\t")
statistics["somesci"]=get_statistics(df)

df = pd.read_csv("datasets/softcite/train-set/softcite-ner/annotations.tsv",delimiter="\t")
statistics["softcite"]=get_statistics(df)

df = pd.read_csv("datasets/benchmark/train-set/benchmark-ner/annotations.tsv",delimiter="\t")
statistics["benchmark"]=get_statistics(df)


intersection = set(statistics["somesci"]["mentions"]).intersection(statistics["softcite"]["mentions"])

print(intersection)
print(len(intersection))