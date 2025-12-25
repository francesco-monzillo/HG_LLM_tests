from openai import OpenAI
import os
import time
import chunking
from nltk.tokenize import sent_tokenize
import json
import requests


gpt_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)


normal_hypergraph_representation = None
time_labeled_hypergraph_representation = None
measure_labeled_hypergraph_representation = None
kg_labeled_hypergraph_representation = None
measure_value_labeled_hypergraph_representation = None

hypergraph_representation_list = [normal_hypergraph_representation, time_labeled_hypergraph_representation, measure_labeled_hypergraph_representation, kg_labeled_hypergraph_representation, measure_value_labeled_hypergraph_representation]
#hypergraph_representation_list = [measure_labeled_hypergraph_representation]

documentation_path = "documentation_and_hierarchy"

with open("../generated_hypergraphs_json/"+ documentation_path +"/test_normal_hypergraph.json", "r") as f:
    hypergraph = json.load(f)
    #hypergraph_representation_list[0] = json.dumps(hypergraph, separators=(',', ':'))
    hypergraph_representation_list[0] = hypergraph

with open("../generated_hypergraphs_json/"+ documentation_path +"/test_time_labeled_hypergraph.json", "r") as f:
    hypergraph = json.load(f)
    #hypergraph_representation_list[1] = json.dumps(hypergraph, separators=(',', ':'))
    hypergraph_representation_list[1] = hypergraph

with open("../generated_hypergraphs_json/"+ documentation_path +"/test_measure_labeled_hypergraph.json", "r") as f:
    hypergraph = json.load(f)
    #hypergraph_representation_list[2] = json.dumps(hypergraph, separators=(',', ':'))
    hypergraph_representation_list[2] = hypergraph

with open("../generated_hypergraphs_json/"+ documentation_path +"/test_kg_labeled_hypergraph.json", "r") as f:
    hypergraph = json.load(f)
    #hypergraph_representation_list[3] = json.dumps(hypergraph, separators=(',', ':'))
    hypergraph_representation_list[3] = hypergraph

with open("../generated_hypergraphs_json/" + documentation_path + "/test_measure_value_labeled_hypergraph.json", "r") as f:
    hypergraph = json.load(f)
    hypergraph_representation_list[4] = hypergraph


vectorstore_wrappers = [[] for _ in range(len(hypergraph_representation_list))]

relevant_chunks = [_ for _ in range(len(hypergraph_representation_list))]

vectorstore_node_wrappers = [[] for _ in range(len(hypergraph_representation_list))]

vectorstore_edge_wrappers = [[] for _ in range(len(hypergraph_representation_list))]

sub_folder = "better_embeddings"

# Load vector stores from local directory
for i, wrapper in enumerate(vectorstore_wrappers):

    node_chunks_parent = None
    hypergraph_parent = None
    edge_chunks_parent = None

    if (sub_folder != ""):
        hypergraph_parent = f"./vectorstores/{documentation_path}/{sub_folder}/vectorstore_{i}"
        node_chunks_parent = f"./vectorstores/{documentation_path}/{sub_folder}/node_chunks/vectorstore_{i}"
        edge_chunks_parent = f"./vectorstores/{documentation_path}/{sub_folder}/edge_chunks/vectorstore_{i}"
    else:
        hypergraph_parent = f"./vectorstores/{documentation_path}/weakened_embeddings/vectorstore_{i}"
        node_chunks_parent = f"./vectorstores/{documentation_path}/weakened_embeddings/node_chunks/vectorstore_{i}"
        edge_chunks_parent = f"./vectorstores/{documentation_path}/weakened_embeddings/edge_chunks/vectorstore_{i}"

    for name in os.listdir(hypergraph_parent):
        path = os.path.join(hypergraph_parent, name)

        if os.path.isdir(path):
            print("Folder:", path)
            vectorstore_wrappers[i] = chunking.returnLocalVectorStoreObject(f"{path}")

    # for name in os.listdir(node_chunks_parent):
    #     path = os.path.join(node_chunks_parent, name)

    #     if os.path.isdir(path):
    #         print("Folder:", path)
    #         vectorstore_node_wrappers[i] = chunking.returnLocalVectorStoreObject(f"{path}")

    # for name in os.listdir(edge_chunks_parent):
    #     path = os.path.join(edge_chunks_parent, name)

    #     if os.path.isdir(path):
    #         print("Folder:", path)
    #         vectorstore_edge_wrappers[i] = chunking.returnLocalVectorStoreObject(f"{path}")





with open("./questions/questions.tsv", "r") as f:
    text_file = f.read()

    for i, line in enumerate(text_file.split("\n")):

        if(i == 0):
            continue

        if line.strip():  # Ensure the line is not empty
        
            if(line[0] == "#"):
                continue

            sentences = sent_tokenize(line)

            print(sentences[0])

            requests = [sentences[1], sentences[2]]

            for request in requests:

                request_string = ""
                            
                if request == sentences[1]:
                    request_string = 1
                else:
                    request_string = 2
    
                print("Request: " + request)

                for i, hypergraph_representation in enumerate(hypergraph_representation_list):
                    good_ending = False

                    while not good_ending:
                        try:

                            to_open_directory = ""
                            internal_folder = ""

                            if i == 0:
                                internal_folder = "Normal"
                            elif i == 1:
                                internal_folder = "Time-labeled"
                            elif i == 2:
                                internal_folder = "Measure-labeled"
                            elif i == 3:
                                internal_folder = "KG-labeled"
                            elif i == 4:
                                internal_folder = "Measure-value-labeled"

                            relevant_chunks[i] = chunking.getRelevantChunks(vectorstore_wrappers[i], request, k=4000)

                            
                            good_ending = True

                            response = gpt_client.chat.completions.create(

                                        model="gpt-5-2025-08-07",

                                        messages=[

                                            {"role": "system", "content": relevant_chunks[i]},

                                            {"role": "user", "content": request}

                                        ],

                                        top_p=1,
                                        temperature= 0,
                                        verbosity="low",
                                        reasoning_effort="medium"

                                    )

                            print(response.choices[0].message)
                            
                            good_ending = True

                            # hg_string = ""
                            # fileName = ""
                            # if(i == 0):
                            #     hg_string = "Normal HG"
                            #     fileName = "normal_hg_responses.txt"
                            # elif(i == 1):
                            #     hg_string = "Time-labeled HG"
                            #     fileName = "time_labeled_hg_responses.txt"
                            # elif(i == 2):
                            #     hg_string = "Measure-labeled HG"
                            #     fileName = "measure_labeled_hg_responses.txt"
                            # elif(i == 3):
                            #     hg_string = "KG-labeled HG"
                            #     fileName = "kg_labeled_hg_responses.txt"
                            # elif(i == 4):
                            #     hg_string = "Measure-value-labeled HG"

                            # print("Response for " + hg_string + ":")
                            # print(response.text)

                            # with open("./responses/" + fileName, "a") as resp_f:
                            #     resp_f.write(f"Question: {sentences[0]}\n")
                            #     resp_f.write(f"Request: {request}\n")
                            #     resp_f.write(f"Response:\n{response.text}\n")
                            #     resp_f.write("--------------------------------------------------\n")
                            

                            # time.sleep(20)
                        except Exception as e:
                            print("timeout...Retry")
                            print(e)
                            # enc = tiktoken.get_encoding("cl100k_base")  # Similar to Gemini tokenizer
                            # text = hypergraph_representation + request
                            # num_tokens = len(enc.encode(text))

                            # print(f"{num_tokens} tokens")
                            time.sleep(20)
            