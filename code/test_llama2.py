from evaluation_llama2 import EvaluationGPT4All

prompt = [
        {"role":"user","content":"I would like you to software mentions in a given text. Use the following delimiters: \"@@\" at the beginning \"##\" at the end. Include the delimiters within the text and do not alter the text."},
        {"role":"user","content":"Here is an input example: \"I am using Microsoft word with the SPSS package\""},
        {"role":"user","content":"The answer should be: \"I am using @@Microsoft word## with the @@SPSS## package\""},
        {"role":"user","content":"Here is another example: \"All of the statistical analyses were conducted using SPSS17.0  (SPSS Inc., Chicago, IL, USA).\”"},
        {"role":"user","content":"The answer should be: \"All of the statistical analyses were conducted using @@SPSS##17.0 (SPSS Inc., Chicago, IL, USA).\""},
        {"role":"user","content":"Here is another example:\"All analyses were performed with SPSS for Windows (SPSS, version 23; Chicago, IL, USA), and p values < 0.05 (two-way) were considered to indicate statistical significance.\""},
        {"role":"user","content":"The answer should be:\"All analyses were performed with @@SPSS## for Windows (@@SPSS##, version 23; Chicago, IL, USA), and p values < 0.05 (two-way) were considered to indicate statistical significance.\""},
        {"role":"user","content":"Here is another example:\"Data were analyzed in Microsoft Excel 10.0 (Microsoft, Redmond, WA) and STATA/IC 12.1 (Stata Corp, College Station, TX).\""},
        {"role":"user","content":"The answer should be:\"Data were analyzed in Microsoft @@Excel## 10.0 (Microsoft, Redmond, WA) and @@STATA##/@@IC## 12.1 (Stata Corp, College Station, TX).\""},
    ]

input = "We used the Statistical Package for Social Sciences (IBM SPSS v. 24.0.0.0) for all analyses."
output = "We used the @@Statistical Package for Social Sciences## (IBM @@SPSS## v. 24.0.0.0) for all analyses."

question = "Can you annotate the following sentence?"

evaluation = EvaluationGPT4All("llama-2-7b-chat","http://localhost:4891/v1","not needed for a local LLM")

evaluation.set_prompt(prompt, question)

tp, fp, fn = evaluation.evaluation(input, output)

print("True positives:"+str(tp))
print("False positives:"+str(tp))
print("False negatives:"+str(tp))