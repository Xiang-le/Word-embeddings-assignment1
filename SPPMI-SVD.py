# Chen Xiang Le A0252434L
# Assignment 1: SPPMI-SVD model

from gensim.models import Word2Vec
import ebooklib
from ebooklib import epub                         # for reading epub files
from bs4 import BeautifulSoup                     # for parsing epub book
import re                                         # for string manipulation
from sklearn.decomposition import PCA             # for dimensionality reduction
import matplotlib.pyplot as plt                   # for plots
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD    # for SPPMI-SVD

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

# Step 3: Build the vocabulary
vocab = {word: idx for idx, word in enumerate(set(word for sentence in cleaned_sentences for word in sentence))}
#print(vocab)

# Step 4: Build the co-occurrence matrix
# Hyperparameters to tune
WINDOW_SIZE = 5 
K = 3

vocab_size = len(vocab)
co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)

for sentence in cleaned_sentences:
    sentence_length = len(sentence)
    for idx, word in enumerate(sentence):
        word_idx = vocab[word]
        
        # Define the context window
        start = max(0, idx - WINDOW_SIZE)
        end = min(sentence_length, idx + WINDOW_SIZE + 1)
        
        # Update co-occurrence counts for words in the window
        for context_idx in range(start, end):
            if idx != context_idx:  # Skip the word itself
                context_word_idx = vocab[sentence[context_idx]]
                co_matrix[word_idx, context_word_idx] += 1
#print(co_matrix)     

# Step 5: Calculate PMI matrix
co_occurrence_sum = np.sum(co_matrix)
p_word = np.sum(co_matrix, axis=1) / co_occurrence_sum
sppmi_matrix = np.zeros_like(co_matrix)

for i in range(vocab_size):
    for j in range(vocab_size):
        if co_matrix[i, j] > 0:
            pmi = np.log((co_matrix[i, j] / co_occurrence_sum) / (p_word[i] * p_word[j]))
            sppmi_matrix[i, j] = max(pmi-np.log(K), 0)  # Positive SPPMI

# Step 8: Apply SVD on the SPPMI matrix
U, Sigma, Vt = np.linalg.svd(sppmi_matrix, full_matrices=False)

# Extract the top-k components
k = 2  # Number of dimensions
U_k = U[:, :k]                # First k columns of U
Sigma_k = np.diag(Sigma[:k])  # Top k singular values as a diagonal matrix
V_k = Vt[:k, :]               # First k rows of V^T

# Compute U_k Sigma_k^{1/2}
Sigma_k_sqrt = np.sqrt(Sigma_k)
pmi_embeddings = U_k @ Sigma_k_sqrt


# Cosine similarity check between 'queen' and 'king'
queen = pmi_embeddings[vocab["queen"]]
king = pmi_embeddings[vocab["king"]]
print((queen@king) / (np.linalg.norm(queen)*np.linalg.norm(king)))

# Visualization
plt.figure(figsize=(10, 8))
plt.scatter(pmi_embeddings[:, 0], pmi_embeddings[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, (pmi_embeddings[i, 0], pmi_embeddings[i, 1]))
plt.title("PMI-SVD Word Embeddings")
plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")
plt.show()
