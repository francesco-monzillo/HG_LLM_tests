import ast
from google import genai
import os
import json
import time
import nltk
import sys
from google.genai import types
#nltk.download('punkt_tab')
import tiktoken
from nltk.tokenize import sent_tokenize
import chunking
import re
import numpy as np

import langchain_google_genai.embeddings as genai_embeddings
embedding_model = genai_embeddings.GoogleGenerativeAIEmbeddings(model="text-embedding-004")

def safe_filename(s: str) -> str:
    # Replace illegal or unsafe characters with underscores
    return re.sub(r'[^a-zA-Z0-9._-]', '_', s)

# Open and read the file

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


chunks_for_each_hypergraph = [_ for _ in range(len(hypergraph_representation_list))]

chunks_for_each_hypergraph_nodes = [_ for _ in range(len(hypergraph_representation_list))]
chunks_for_each_hypergraph_edges = [_ for _ in range(len(hypergraph_representation_list))]


for i, hypergraph_representation in enumerate(hypergraph_representation_list):

    nolabel = False

    #Because normal hypergraph does not have univocal identifiers for edges (first presented in this list)
    if(i == 0):
        nolabel = True

    chunks = chunking.create_chunks_from_edges_and_nodes(hypergraph_representation["nodes_contained_for_each_edge"], hypergraph_representation["node-data"], nolabel)
    chunks_for_each_hypergraph[i] = chunks

    chunks_for_each_hypergraph_nodes[i] = chunking.create_chunks_from_nodes(hypergraph_representation["node-data"])
    chunks_for_each_hypergraph_edges[i] = chunking.create_chunks_from_edges(hypergraph_representation["nodes_contained_for_each_edge"])


vectorstore_wrappers = [[] for _ in range(len(hypergraph_representation_list))]

relevant_chunks = [_ for _ in range(len(hypergraph_representation_list))]

vectorstore_node_wrappers = [[] for _ in range(len(hypergraph_representation_list))]

vectorstore_edge_wrappers = [[] for _ in range(len(hypergraph_representation_list))]

sub_folder = "better_embeddings"

# Save vector stores locally
# for i, wrapper in enumerate(vectorstore_wrappers):

#     if (sub_folder != ""):
#         vectorstore_wrappers[i] = chunking.buildVectorStoreObject(chunks_for_each_hypergraph[i], embeddings= "GEMINI", maximum_tokens_for_embedding_request = 2048)
#     else:
#         vectorstore_wrappers[i] = chunking.buildVectorStoreObject(chunks_for_each_hypergraph[i], embeddings= "GEMINI")
#     #for j, vectorstore in enumerate(vectorstore_wrappers[i]):
#     if (sub_folder != ""):
#         vectorstore_wrappers[i].save_local(f"./vectorstores/{documentation_path}/{sub_folder}/vectorstore_{i}/part_1")
#     else:
#         vectorstore_wrappers[i].save_local(f"./vectorstores/{documentation_path}/weakened_embeddings/vectorstore_{i}/part_1")


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




#Continue From here... I will need to compare each chunk embedding with the question  embedding through cosine similarity. Then, select the top k chunks to provide as context to Gemini.

client = genai.Client(
    api_key= os.environ.get("GEMINI_API")
)

#print(hypergraph_representation)

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


                            ################################################

                            # # #Open file containing node chunks relevant to the question
                            # to_open_node_directory = ""

                            # if (sub_folder != ""):
                            #     to_open_node_directory = f"./RELEVANT_CHUNKS_FOR_EACH_QUESTION/{documentation_path}/{sub_folder}/{internal_folder}/node_chunks/relevant_chunks_{i}_question_{sentences[0][0:len(sentences[0]) - 1]}_request_{request_string}.txt"
                            # else:
                            #     to_open_node_directory = f"./RELEVANT_CHUNKS_FOR_EACH_QUESTION/{documentation_path}/weakened_embeddings/{internal_folder}/node_chunks/relevant_chunks_{i}_question_{sentences[0][0:len(sentences[0]) - 1]}_request_{request_string}.txt"

                            # node_relevant_chunks = None

                            # #Open file containing node chunks relevant to the question
                            # with open(to_open_node_directory, "r") as node_chunks_file:
                            #     node_relevant_chunks = node_chunks_file.read()

                            # # Chunks most similar to the node chunks relevant to the question
                            # relevant_chunks[i] = chunking.getRelevantChunks(vectorstore_edge_wrappers[i], request, k=2000)

                            # # Embed the node chunks relevant to the question
                            # request_embedding = embedding_model.embed_query(request)

                            # # Convert to numpy array (right format for FAISS) and transform to 2D vector
                            # request_embedding = np.array(request_embedding, dtype=np.float32).reshape(1, -1)

                            # # Build vector store from those chunks through their embeddings
                            # indexes, documents = vectorstore_edge_wrappers[i].index.search(request_embedding, k=2000)

                            # relevant_embeddings = np.vstack([vectorstore_edge_wrappers[i].index.reconstruct(int(index)) for index in indexes[0]])

                            # # Build vector store from those chunks
                            # vectorstore_edge_wrappers[i] = chunking.buildFromEmbeddingsAndDocuments(relevant_embeddings, relevant_chunks[i], embedding_model = embedding_model)

                            # #Add relevant nodes chunks to the vector store before analyzing semantic similarity with the question
                            # vectorstore_edge_wrappers[i].add_texts(node_relevant_chunks.split("\n"))

                            # relevant_chunks[i] = chunking.getRelevantChunks(vectorstore_edge_wrappers[i], request, k=1000)

                            ################################################

                            #I executed tests with GEMINI 2.5 PRO with k = 4000

                            number_of_relevant_chunks = 1000
                            relevant_chunks[i] = chunking.getRelevantChunks(vectorstore_wrappers[i], request, k = number_of_relevant_chunks)

                            stop = False

                            while not stop:

                                number_of_relevant_chunks += 50

                                if(chunking.count_tokens_in_chunks(relevant_chunks[i]) < 70000):

                                    futureRelevantChunks = chunking.getRelevantChunks(vectorstore_wrappers[i], request, k = number_of_relevant_chunks)

                                    if(chunking.count_tokens_in_chunks(futureRelevantChunks) >= 70000):
                                        stop = True
                                    else:
                                        relevant_chunks[i] = futureRelevantChunks
                                else:
                                    stop = True
                                
                            if(sub_folder != ""):
                                to_open_directory = f"./RELEVANT_CHUNKS_FOR_EACH_QUESTION/{documentation_path}/{sub_folder}/{internal_folder}/relevant_chunks_{i}_question_{sentences[0][0:len(sentences[0]) - 1]}_request_{request_string}.txt"
                            else:
                                to_open_directory = f"./RELEVANT_CHUNKS_FOR_EACH_QUESTION/{documentation_path}/weakened_embeddings/{internal_folder}/relevant_chunks_{i}_question_{sentences[0][0:len(sentences[0]) - 1]}_request_{request_string}.txt"


                            #relevant_chunks[i] = "\n".join([chunk for chunk in relevant_chunks[i]])

                            with open(to_open_directory, "w") as rc_f:

                                for chunk in relevant_chunks[i]:
                                    rc_f.write(f"{chunk}")

                                    # ast.literal_eval to convert string representation of dict back to dict
                                    # try:
                                    #     rc_f.write(json.dumps(ast.literal_eval(chunk), indent= 2))
                                    # except (ValueError, SyntaxError) as e:
                                    #     #print(f"Printing chunk without eval due to error: {e}")
                                    #     rc_f.write(chunk)

                                    # rc_f.write("\n")

                            good_ending = True

                            # response = client.models.generate_content(
                            #             model="gemini-2.5-pro",
                            #             contents=[relevant_chunks[i], request],
                            #             Deterministic output
                            #             config = types.GenerateContentConfig(
                            #             temperature=0.0,
                            #             seed=42,             # Use a fixed integer seed
                            #             top_p=1.0,           # Include all tokens for sampling (though T=0.0 dominates)
                            #             top_k=1,             # Only consider the single most probable token
                            #         )
                            # )
                            # good_ending = True

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
            