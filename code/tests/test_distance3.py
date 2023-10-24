import jellyfish

# Example lists
list1 = ["01010", "11000", "10101", "11110"]
list2 = ["01100", "10001", "10100", "11011"]

# Hamming distance threshold
threshold = 2

for elem in list1:
    print(elem)
    filtered_elements = [elem2 for elem2 in list2 if jellyfish.hamming_distance(elem, elem2) < threshold]
    print(filtered_elements)
    print("------------------")

# Filter elements based on Hamming distance using jellyfish
filtered_elements = [elem for elem in list1 if all(jellyfish.hamming_distance(elem, elem2) > threshold for elem2 in list2)]

print(filtered_elements)