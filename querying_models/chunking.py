import json
import os
import time
from google import genai
from langchain_community.vectorstores import FAISS
import sys
import faiss
import uuid
import tiktoken
import numpy as np


def count_tokens_in_chunks(chunks):
    enc = tiktoken.get_encoding("cl100k_base")  # Similar to Gemini tokenizer
    total_tokens = 0
    for chunk in chunks:
        total_tokens += len(enc.encode(chunk))
    return total_tokens

def create_chunks_from_edges_and_nodes(edge_list, node_dict, nolabel = False):
    chunks = []

    for edge_dict in edge_list:
        for edge_id, nodes in edge_dict.items():
            edge = None
            #Not using univocal identifier for edges in case of normal hypergraph
            if(nolabel):
                edge = nodes
            else:
                edge = {edge_id: nodes}

            chunks.append(json.dumps(edge, separators=(',', ':')))

    for node_id, properties in node_dict.items():
        node = {node_id: properties}
        chunks.append(json.dumps(node, separators=(',', ':')))

    return chunks


def create_chunks_from_edges(edge_list):
    chunks = []

    for edge_dict in edge_list:
        for edge_id, nodes in edge_dict.items():
            edge = {edge_id: nodes}
            chunks.append(json.dumps(edge, separators=(',', ':')))

    return chunks

def create_chunks_from_nodes(node_dict):
    chunks = []

    for node_id, properties in node_dict.items():
        node = {node_id: properties}
        chunks.append(json.dumps(node, separators=(',', ':')))

    return chunks

def return_chunks_elements(text, whole_chunk = False):

    if whole_chunk:
        return [text]

    try:
        chunk = json.loads(text)

        to_return_elements = []

        for key, value in chunk.items():
            try:
                key = int(key)

                if isinstance(value, list):
                    for item in value:
                        to_return_elements.append(item) 
                    continue
                else:
                    for property, content in value.items():
                        to_return_elements.append(property + ": " + str(content))

            except:

                to_return_elements.append(key)

                if isinstance(value, list):
                    for item in value:
                        to_return_elements.append(item) 
                    continue
                else:
                    for property, content in value.items():
                        to_return_elements.append(property + ": " + str(content))
                    
                    print(to_return_elements[-1])
    except json.JSONDecodeError:
        to_return_elements = [text]

    return to_return_elements

def add_embedding_for_chunks(embedding_model, chunks):
    embeddings = []
    array_of_elements = []
    for chunk in chunks:
        #True means that the whole chunk is considered as a single element for embedding
        array_of_elements.append(return_chunks_elements(chunk, True))
        

    chunks_embeddings = None
    while chunks_embeddings is None:
        try:
            chunks_embeddings = np.array(embedding_model.embed_documents([element for array in array_of_elements for element in array]))
        except Exception as e:
            print(f"Error while embedding chunks: {e}")
            time.sleep(20)

    # Aggregate the embeddings (e.g., by averaging)
    # Also, converting to float32 as FAISS requires that datatype

    for array in array_of_elements:

        #Retrieve only the embeddings related to the current chunk
        interested_embeddings = chunks_embeddings[:len(array)]
        #Remove the used embeddings from the main array
        chunks_embeddings = chunks_embeddings[len(array):]

        aggregate_embedding = np.mean(interested_embeddings, axis=0).astype(np.float32)

        # Normalize the aggregate embedding
        # when using L2 distance, as it makes it equivalent to cosine similarity.
        faiss.normalize_L2(aggregate_embedding.reshape(1, -1))

        #Reshaping to a matrix with one row and one column for each dimension (FOR FAISS)
        embeddings.append(aggregate_embedding)

    return embeddings


def buildFromEmbeddingsAndDocuments(embeddings, documents, embedding_model = None):

    if embedding_model is None:
        import langchain_google_genai.embeddings as genai_embeddings
        embedding_model = genai_embeddings.GoogleGenerativeAIEmbeddings(model="text-embedding-004")

    return FAISS.from_embeddings([(document, embedding) for document, embedding in zip(documents, embeddings)], embedding_model)


inserted_docs = dict()


def incorporate_texts_in_vectorstore(vectorstore_wrapper, texts, embedding_for_each_document, embedding_model):

    if vectorstore_wrapper is None:
        
        #for document in embedding_for_each_document:
        #    faiss.normalize_L2(document.reshape(1, -1))
        vectorstore_wrapper = FAISS.from_embeddings([(text, embedding) for text, embedding in zip(texts, embedding_for_each_document)], embedding_model)

    else:
        for t in texts:
            if not t or not t.strip():
                continue
        
        vectorstore_wrapper.add_embeddings([(text, embedding) for text, embedding in zip(texts, embedding_for_each_document)])

    return vectorstore_wrapper


def buildVectorStoreObject(chunks, embeddings = "GEMINI", maximum_tokens_for_embedding_request = None):
    
    vectorstore_wrapper = None

    if embeddings == "OPENAI":
        from langchain_openai import OpenAIEmbeddings
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        if maximum_tokens_for_embedding_request is None:
            maximum_tokens_for_embedding_request = 8191
    else:
        import langchain_google_genai.embeddings as genai_embeddings

        embedding_model = genai_embeddings.GoogleGenerativeAIEmbeddings(model="text-embedding-004")
        # client = genai.Client(api_key=os.getenv("GEMINI_API"))
        # models = client.models.list()
        # for m in models:
        #     print(m.name, m.supported_actions)

        # sys.exit(0)
        if maximum_tokens_for_embedding_request is None:
            maximum_tokens_for_embedding_request = 30000

    number_of_tokens = 0

    enc = tiktoken.get_encoding("cl100k_base")  # Similar to Gemini tokenizer

    for chunk in chunks:
        text = chunk
        number_of_tokens += len(enc.encode(text))

    print(f"Number of tokens: {number_of_tokens}")

    if(number_of_tokens > maximum_tokens_for_embedding_request):
        print("Warning: The total number of tokens in chunks exceeds the maximum allowed for the embedding model... Chunking on the chunks is required.")

        chunks_per_request = []
        embeddings_for_chunks = []

        segmented_chunks = []

        current_token_count = 0

        #Counter
        i = 0

        while i < len(chunks):

            chunk = chunks[i]
            chunk_token_count = len(enc.encode(chunk))
            if current_token_count + chunk_token_count <= maximum_tokens_for_embedding_request:
                chunks_per_request.append(chunk)

                current_token_count += chunk_token_count
            else:
                current_token_count = 0
                
                #embeddings_for_chunks = add_embedding_for_chunks(embedding_model, chunks_per_request)
            
                segmented_chunks.append(chunks_per_request)
                
                #vectorstore_wrapper = incorporate_texts_in_vectorstore(vectorstore_wrapper, chunks_per_request, embeddings_for_chunks, embedding_model)

                print(f"Created vectorstore with {len(chunks_per_request)} chunks.")

                chunks_per_request = [chunk]
                current_token_count  = chunk_token_count

                
                if(len(enc.encode(chunk)) > maximum_tokens_for_embedding_request):
                    print("Error: A single chunk exceeds the absolute maximum token limit for embeddings.")
                    decomposition = []
                    for offset in range(0, len(chunk), maximum_tokens_for_embedding_request):
                        #Advance by maximum_tokens_for_embedding_request tokens
                        to_add = maximum_tokens_for_embedding_request

                        if(offset + maximum_tokens_for_embedding_request > len(chunk)):
                            to_add = len(chunk) - offset

                        decomposition.append(chunk[offset : offset + to_add])
                    
                    for j, chunk in enumerate(decomposition):
                        if (j == 0):
                            continue

                        chunks.insert(i + j, chunk)

                    #Add the first section of the decomposed chunk    
                    chunks_per_request = [decomposition[0]]

                    # embedding_for_chunk = np.array(embedding_model.embed_query(decomposition[0]))
        
                    # # Normalize the embedding
                    # # when using L2 distance, as it makes it equivalent to cosine similarity.
                    # faiss.normalize_L2(embedding_for_chunk.reshape(1, -1).astype(np.float32))

                    # embeddings_for_chunks = [embedding_for_chunk]

                    segmented_chunks.append(chunks_per_request)
                    #vectorstore_wrapper = incorporate_texts_in_vectorstore(vectorstore_wrapper, chunks_per_request, embeddings_for_chunks, embedding_model)

                    chunks_per_request = []
                    embeddings_for_chunks = []
                    current_token_count = 0

                #In this case, reset counter and empty list 'cause I just directly inserted the new vectorstore in the wrapper list                 

            i += 1
            
        if(len(chunks_per_request) > 0):
            #In this case, the remaining chunk can be added to the vectorstore through the aggregated embedding (otherwise it would have been added already in the else case above)
            # embeddings_for_chunks = add_embedding_for_chunks(embedding_model, chunks_per_request)
            segmented_chunks.append(chunks_per_request)

            #vectorstore_wrapper = incorporate_texts_in_vectorstore(vectorstore_wrapper, chunks_per_request, embeddings_for_chunks, embedding_model)

            print(f"Created vectorstore with {len(chunks_per_request)} chunks.")

        total_length = len(segmented_chunks)

        # Reset counter
        i = 0
        while i < len(segmented_chunks):

            chunks_per_request = []

            j = 0

            while j < 9 and (i + j) < len(segmented_chunks):
                if(i + j < len(segmented_chunks)):
                    chunks_per_request += segmented_chunks[i + j]

                j += 1

            embeddings_for_chunks = add_embedding_for_chunks(embedding_model, chunks_per_request)
            
            #embeddings_for_chunks = document_embeddings_couples[i][1].append(document_embeddings_couples[i+1][1]).append(document_embeddings_couples[i+2][1])

            vectorstore_wrapper = incorporate_texts_in_vectorstore(vectorstore_wrapper, chunks_per_request, embeddings_for_chunks, embedding_model)

            i = i + j

            print(f"Combined vectorstore with chunks from segmented chunk sets. Progress: {i}/{total_length}")


    return vectorstore_wrapper


def getRelevantChunks(wrappers, question, k=1500):

    relevant_chunks = []
    import math

    relevant_chunks = wrappers.similarity_search(question, k = k)   

    return [chunk.page_content for chunk in relevant_chunks]

def returnLocalVectorStoreObject(file_path, embedding_model = "GEMINI"):
    vectorstore_wrapper = []

    if not os.path.exists(file_path):
        print(f"Error: The specified vectorstore file {file_path} does not exist.")
        sys.exit(1)

    embedding_model = None

    if embedding_model == "OPENAI":
        from langchain_openai import OpenAIEmbeddings
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        import langchain_google_genai.embeddings as genai_embeddings
        embedding_model = genai_embeddings.GoogleGenerativeAIEmbeddings(model="text-embedding-004")

    vectorstore_wrapper = FAISS.load_local(file_path, embedding_model, allow_dangerous_deserialization = True)

    return vectorstore_wrapper
