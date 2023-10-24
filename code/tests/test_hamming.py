import jellyfish

def hamming_distance(str1, str2):
    # Calculate the Hamming distance between two strings of equal length
    if len(str1) != len(str2):
        raise ValueError("Input strings must have the same length")
    
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

corpus = ["SPSS"]
predictions = ["ANOVA", "SSS"]

print("Hamming")

threshold = 2

filtered_elements = [elem for elem in corpus if all(jellyfish.hamming_distance(elem, elem2) > threshold for elem2 in predictions)]
print (filtered_elements)

tp = [elem for elem in corpus if all(jellyfish.levenshtein_distance(elem, elem2) > threshold for elem2 in predictions)]
print (tp)


print("---------------")