#from openai import OpenAI

#client = OpenAI()

#sentence = "Parallel computing is the key to scalability."

#response = client.embeddings.create(
#    model="text-embedding-3-large",
#    input=sentence
#)

#embedding_vector = response.data[0].embedding
#print(f"Length: {len(embedding_vector)}")
#print(embedding_vector[:10])  # First 10 values


#Embedding an answer written in natural language
from sentence_transformers import SentenceTransformer

# Load the model (downloads on first run)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Your sentence
sentence = "The most influent knowledge graph on 2022/07/19 is 0083-9926"

# Generate embedding
embedding = model.encode(sentence)

print(f"Embedding length: {len(embedding)}")
print(embedding[:10])  # First 10 values


#Extract entities from the same sentence?
import spacy

#Download model
import spacy.cli
spacy.cli.download("en_core_web_trf")
#

nlp = spacy.load("en_core_web_trf")
doc = nlp("The most influent knowledge graph on 2022/07/19 is 0083-9926")
for ent in doc.ents:
    print(ent.text, ent.label_)