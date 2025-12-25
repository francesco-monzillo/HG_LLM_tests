from google import genai
import os
import sys
import json
import time
import nltk
from openai import OpenAI
#nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize

# Open and read the file
with open("../generated_hypergraphs_json/test_time_labeled_hypergraph.json", "r") as f:
    hypergraph = json.load(f)

client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

hypergraph_representation = json.dumps(hypergraph, separators=(',', ':'))

print(hypergraph_representation)

sys.exit(0)

response = client.chat.completions.create(
    model="deepseek-r1",
    messages=[
        {"role": "system", "content": hypergraph_representation},
        {"role": "user", "content": "hi"}
    ]
)

print(response.text)

sys.exit(0)

with open("./questions/questions.tsv", "r") as f:
    text_file = f.read()

    for i, line in enumerate(text_file.split("\n")):

        if(i == 0):
            continue

        if line.strip():  # Ensure the line is not empty
        
            sentences = sent_tokenize(line)

            print(sentences[0])

            requests = [sentences[1], sentences[2]]

            for request in requests:
                good_ending = False

                while not good_ending:
                    try:
                        response = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=[json.dumps(hypergraph), request]
                        )
                        good_ending = True

                        print(response.text)

                    except Exception as e:
                        print("timeout...Retry")
                        time.sleep(10)
            