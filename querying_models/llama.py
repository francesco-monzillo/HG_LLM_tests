from platform import processor
from openai import OpenAI
import os
import time
import chunking
from nltk.tokenize import sent_tokenize
import json
import lmstudio as lms

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Llama 3.2 1B Instruct GGUF model
#llama_client = lms.llm("qwen3_vl_2b", config={"contextLength": 130000, "seed": 42})


torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)


model_id = "Qwen/Qwen2-1.5B"
hugging_face_token = os.getenv("HUGGINGFACE_HG_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_auth_token = hugging_face_token)

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map = "auto",   # auto GPU/CPU distribution
    max_memory={
        0: "10GB",   # GPU memory
        "cpu": "32GB"  # CPU offload
    },
    offload_folder="offload",
    load_in_4bit=True,  # load the model in 4-bit precision
    torch_dtype=torch.float16,
    trust_remote_code=True,
    use_auth_token = hugging_face_token
)

normal_hypergraph_representation = None
time_labeled_hypergraph_representation = None
measure_labeled_hypergraph_representation = None
kg_labeled_hypergraph_representation = None
measure_value_labeled_hypergraph_representation = None

hypergraph_representation_list = [normal_hypergraph_representation, time_labeled_hypergraph_representation, measure_labeled_hypergraph_representation, kg_labeled_hypergraph_representation, measure_value_labeled_hypergraph_representation]

documentation_path = "documentation_and_hierarchy"

# with open("../generated_hypergraphs_json/"+ documentation_path +"/test_normal_hypergraph.json", "r") as f:
#     hypergraph = json.load(f)
#     #hypergraph_representation_list[0] = json.dumps(hypergraph, separators=(',', ':'))
#     hypergraph_representation_list[0] = hypergraph

# with open("../generated_hypergraphs_json/"+ documentation_path +"/test_time_labeled_hypergraph.json", "r") as f:
#     hypergraph = json.load(f)
#     #hypergraph_representation_list[1] = json.dumps(hypergraph, separators=(',', ':'))
#     hypergraph_representation_list[1] = hypergraph

# with open("../generated_hypergraphs_json/"+ documentation_path +"/test_measure_labeled_hypergraph.json", "r") as f:
#     hypergraph = json.load(f)
#     #hypergraph_representation_list[2] = json.dumps(hypergraph, separators=(',', ':'))
#     hypergraph_representation_list[2] = hypergraph

# with open("../generated_hypergraphs_json/"+ documentation_path +"/test_kg_labeled_hypergraph.json", "r") as f:
#     hypergraph = json.load(f)
#     #hypergraph_representation_list[3] = json.dumps(hypergraph, separators=(',', ':'))
#     hypergraph_representation_list[3] = hypergraph

# with open("../generated_hypergraphs_json/" + documentation_path + "/test_measure_value_labeled_hypergraph.json", "r") as f:
#     hypergraph = json.load(f)
#     hypergraph_representation_list[4] = hypergraph


vectorstore_wrappers = [[] for _ in range(len(hypergraph_representation_list))]

relevant_chunks = [_ for _ in range(len(hypergraph_representation_list))]


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

            requests_for_llm = [sentences[1], sentences[2]]

            for request in requests_for_llm:

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

                            relevant_chunks_file_path = f"./RELEVANT_CHUNKS_FOR_EACH_QUESTION/{documentation_path}/better_embeddings/{internal_folder}/relevant_chunks_{i}_question_{sentences[0][0:len(sentences[0]) - 1]}_request_{request_string}.txt"



                            relevant_chunks[i] = None
                            
                            with open(relevant_chunks_file_path, "r") as rel_chunks_file:
                                relevant_chunks[i] = rel_chunks_file.read()


                            # Example text prompt
                            prompt = f"{request} {relevant_chunks[i]}"

                            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to("cuda")
                            
                            # Get the length of the input tokens
                            input_length = inputs.input_ids.shape[1]

                            # Generate the full sequence
                            full_output = model.generate(
                                **inputs,
                                top_p=1.0,
                                do_sample=False,
                            )

                            # 3. Slice the output: Keep only tokens from input_length onwards
                            generated_tokens = full_output[0][input_length:]

                            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

                            print("Response:\n" + response)

                            #response = llama_client.respond(f"{request} {relevant_chunks[i]}", config={"temperature" : 0.0, "topP" : 1})
                            #print(response)



                            response_path = f"./responses/Qwen3-VL-2B/with_{documentation_path}/{internal_folder}_hg_responses.txt"

                            with open(response_path, "a") as response_file:
                                response_file.write(f"Question: {sentences[0]}\nRequest: {request}\nExpected Response: {sentences[3]}\nResponse:\n{response}\n\n")
                                response_file.write("Vote:\n--------------------------------------------------\n")
                            
                            good_ending = True

                        except Exception as e:
                            print("timeout...Retry")
                            print(e)
                            # enc = tiktoken.get_encoding("cl100k_base")  # Similar to Gemini tokenizer
                            # text = hypergraph_representation + request
                            # num_tokens = len(enc.encode(text))

                            # print(f"{num_tokens} tokens")
                            time.sleep(20)
            