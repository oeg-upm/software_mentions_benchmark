import openai
import json

class EvaluationGPT4All:

    model = ""
    endpoint = ""
    api = ""
    instructions = []
    question = ""
    openai.api_base = "http://localhost:4891/v1"

    openai.api_key = "not needed for a local LLM"

    def __init__(self, model, endpoint, api):
        self.model = model
        self.endpoint = endpoint
        self.api = api

    def set_prompt(self, prompt, question):
        self.instructions = prompt
        self.question = question
        

    def __extract_response(self,text):
        result = ""
        response_text = text["choices"][0]["message"]["content"]
        if response_text.find("using \"@@\"") > -1:

            response_text = response_text[response_text.find("using \"@@\"")+12::]
            result = response_text
        else:
            print("No response detected")

        return result
    
    def __extract_annotations(self,text):
        list_annotations = []

        while (len(text) > 0):

            begin = text.find("@@")
            end = text.find("##")        

            if begin > -1 and end > -1:
                list_annotations.append(text[begin+2:end])
                text=text[end+2::]
            else:
                break

        return list_annotations

    def evaluation(self, input, corpus):

        false_negatives = 0
        true_positives = 0
        false_positives = 0

        prompt = self.instructions

        prompt.append({"role":"user","content":"Text:"+input.replace("\n","")})
        prompt.append({"role":"user","content":self.question})

        response = openai.ChatCompletion.create(
            model = self.model,
            messages = prompt,
            max_tokens=250,
            temperature=0,
            top_p=0.95,
            n=1,
            echo=True,
            stream=False,
            reload=True
        )

        print("Raw response:"+str(response["choices"][0]["message"]["content"]))
        results_raw = self.__extract_response(response)
        print("Response:"+str(response))

        list_corpus = self.__extract_annotations(corpus)
        list_predictions = self.__extract_annotations(results_raw)

        if len(list_corpus)<len(list_predictions): 
            max = len(list_corpus) 
        else: 
            max = len(list_predictions)

        for i in range(0,max):
            if list_corpus[i] == list_predictions[i]:
                true_positives = true_positives + 1
            else:
                false_positives = false_positives + 1

        false_negatives = max-(true_positives+false_positives)

        return true_positives, false_positives, false_negatives
                


