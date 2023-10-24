import unittest
import jellyfish

class TestDistances(unittest.TestCase):

    corpus = ['Statistical Package','SSS','Excel']
    predictions = ['IBM Statistical Package', 'SSS', 'STAT']

    def test_exactMatch(self):

        #TRUE POSITIVES
        result_tp = [x for x in self.corpus if x in self.predictions]
                
        #FALSE NEGATIVES
        result_fn = [x for x in self.corpus if x not in self.predictions]
                
        #FALSE POSITIVES
        result_fp = [x for x in self.predictions if x not in self.corpus]

        self.assertEqual(len(result_tp),1)

        self.assertEqual(len(result_fp),1)

        self.assertEqual(len(result_fn),1)


        print("True positives:"+str(result_tp))
        print("False positives:"+str(result_fp))
        print("False negatives:"+str(result_fn))

    def test_Substring(self):

        #TRUE POSITIVES
        #result_tp = [x for x in self.corpus if x.index(substring) > 0 substring for substring in self.predictions ]
        result_tp = [x for x in self.corpus for substring in self.predictions if substring.find(x) >= 0]
        '''
        result_tp = []
        for x in self.corpus:
            for substring in self.predictions:
                if substring.find(x) >= 0:
                    result_tp.append(x)
        '''
                
        #FALSE NEGATIVES
        result_fn = [x for x in self.corpus for substring in self.predictions if substring.find(x) < 0]
                
        #FALSE POSITIVES
        result_fp = [x for x in self.predictions for substring in self.corpus if substring.find(x) < 0]

        result_tp = []
        result_fp = []
        result_fn = []

        string_founded = False

        for x in self.corpus:
            for substring in self.predictions:
                if substring.find(x) >= 0:
                    result_tp.append(x)
                    string_founded = True
            if not string_founded:
                result_fn.append(x)
            else:
                string_founded = False

                string_founded = False

        for x in self.predictions:
            for substring in self.corpus:
                if x.find(substring) >= 0:
                    string_founded = True
            if not string_founded:
                result_fp.append(x)
            else:
                string_founded = False

        self.assertEqual(len(result_tp),1)

        self.assertEqual(len(result_fp),1)

        self.assertEqual(len(result_fn),1)


        print("True positives:"+str(result_tp))
        print("False positives:"+str(result_fp))
        print("False negatives:"+str(result_fn))



if __name__ == '__main__':
    unittest.main()