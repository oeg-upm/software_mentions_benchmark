import unittest
import jellyfish

class TestDistances(unittest.TestCase):

    corpus = ['SPSS']
    predictions = ['ANOVA', 'SSS']

    def test_exactMatch(self):

        #TRUE POSITIVES
        result_tp = [x for x in self.corpus if x in self.predictions]
                
        #FALSE NEGATIVES
        result_fn = [x for x in self.corpus if x not in self.predictions]
                
        #FALSE POSITIVES
        result_fp = [x for x in self.predictions if x not in self.corpus]

        self.assertEqual(len(result_tp),0)

        self.assertEqual(len(result_fp),2)

        self.assertEqual(len(result_fn),1)


        print("True positives:"+str(result_tp))
        print("False positives:"+str(result_fp))
        print("False negatives:"+str(result_fn))

    def test_Hamming(self):

        print("Hamming")

        threshold = 2

        filtered_elements = [elem for elem in self.corpus if all(jellyfish.hamming_distance(elem, elem2) > threshold for elem2 in self.predictions)]
        print (filtered_elements)
        print("---------------")

        '''
        #TRUE POSITIVES
        result_tp = [x for x in self.corpus if x in self.predictions]
                
        #FALSE NEGATIVES
        result_fn = [x for x in self.corpus if x not in self.predictions]
                
        #FALSE POSITIVES
        result_fp = [x for x in self.predictions if x not in self.corpus]

        self.assertEqual(len(result_tp),0)

        self.assertEqual(len(result_fp),2)

        self.assertEqual(len(result_fn),1)

        print("True positives:"+str(result_tp))
        print("False positives:"+str(result_fp))
        print("False negatives:"+str(result_fn))
        '''



if __name__ == '__main__':
    unittest.main()