from test_somesci_model import TestBinModel

test = TestBinModel()
summary = test.eval_text("somesci_files/PMC1635254.ann","somesci_files/PMC1635254.ann")

print (summary)