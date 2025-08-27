# Chen Xiang Le A0252434L
# Assignment 1: Skipgram model

from gensim.models import Word2Vec
import ebooklib
from ebooklib import epub                  # for reading epub files
from bs4 import BeautifulSoup              # for parsing epub book
import re                                  # for string manipulation
from sklearn.decomposition import PCA      # for dimensionality reduction
import matplotlib.pyplot as plt            # for plots

# Step 1: Retrieve the text file
book = epub.read_epub("pg11.epub")
text = []
for item in book.get_items():
    if item.get_type() == ebooklib.ITEM_DOCUMENT:
        soup = BeautifulSoup(item.get_content(), 'html.parser')
        words = soup.get_text()
        text.append(words)
text.pop(0) # remove the pg-header
text.pop(13) # remove the coverpage-wrapper
text.pop(12) # remove the pg-footer
#print(text[11])

# Step 2: Convert the text into a list of sentences [sentence1, sentence2,...] 
# where each sentence is a list of words similar to the example in lecture
'''
sentences = [
    ["I", "love", "natural", "language", "processing"],
    ["Word2Vec", "is", "a", "popular", "model", "for", "embeddings"],
    ["We", "can", "learn", "word", "relationships"],
    ["Word", "embeddings", "capture", "semantic", "meanings"],
    ["I", "enjoy", "teaching", "Word2Vec", "to", "students"]
]
'''

sentences = []
for chapter in text:
    delimiters = r"[.!?]+"
    x = re.split(delimiters, chapter) # split chapter into a list of sentences
    for i in range(1,len(x)): # skip first sentence which is header "Chapter [roman_numeral]" 
        if '*' in x[i]: continue # subchapters are separated by "*  *  *" and needs to be removed
        cleaned_text = x[i].replace("\n"," ")
        cleaned_text = cleaned_text.lower() # make lowercase
        cleaned_text = re.sub(r'[^\w\s]', "", cleaned_text) # remove all punctuations
        cleaned_text = cleaned_text.strip() # remove trailing whitespace
        sentences.append(cleaned_text)
#print(sentences[:3]) # examine the first 3 sentences

cleaned_sentences = []
for sentence in sentences: # split sentence into a list of words
    cleaned_sentences.append(sentence.split())
#print(cleaned_sentences[:3])

# Step 3: Train a Word2Vec model
# Hyperparameters to tune
MIN_COUNT = 5
WINDOW = 5
VECTOR_SIZE = 50

model = Word2Vec(cleaned_sentences, vector_size=VECTOR_SIZE, window=WINDOW, min_count=MIN_COUNT, sg=1)

# View vocabulary in model
vocab = model.wv.index_to_key
#print("Vocabulary in the model:")
#print(list(model.wv.index_to_key))

# View word embedding of 'queen' 
#print("\nVector representation of the word 'queen':")
#print(model.wv['queen'])

# Find top 5 words that have similar embedding to 'queen'
print("\nWords most similar to 'queen':")
print(model.wv.most_similar("queen", topn=5))
print("Cosine similarity between 'king' and 'queen': ", model.wv.similarity('queen', 'king'))

# Test to see if the word embeddings are able to differentiate between human and animal
print(model.wv.doesnt_match(["king", "queen", "turtle", "alice"]))


# Visualization
word_embeddings = [model.wv[word] for word in vocab]
#print(word_embeddings)
pca = PCA(n_components=2, random_state=42)
reduced_embeddings = pca.fit_transform(word_embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])

for i, word in enumerate(words):
    plt.annotate(word, xy=(reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
plt.title("Word Embeddings Visualization after PCA")
plt.show()