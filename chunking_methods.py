from utils import word_tokenization, sent_tokenization, split_by_separator, get_embedding, get_clutering_output
from sentence_transformers import util
import numpy as np
import regex as re

# Fixed Size Chunking: split the text by whitespace / nltk word tokenizer / spacy text splitter / gensim word tokenizer
def fixed_size_chunking(text, nlp_pipeline='None', splitting_method='whitespace', chunk_size=200, overlap=0):
    # Split the text into tokens
    if splitting_method == 'whitespace':
        # Split the text into tokens based on whitespace
        tokens = text.split()
    else:
        # Split the text into tokens based on the selected method
        tokens = word_tokenization(text, nlp_pipeline, splitting_method)

    chunks = []
    start = 0

    # Loop through the tokens with the specified step size
    while start < len(tokens):
        # Determine the end index of the current chunk
        end = min(start + chunk_size, len(tokens))
        # Join the tokens back into a string for this chunk
        chunk = ' '.join(tokens[start:end])
        # Add the chunk to the result list
        chunks.append(chunk)
        # Move the window forward, subtracting overlap to allow token reuse
        start += chunk_size - overlap

    return chunks


# Recursive Chunking
def recursive_chunking(text, chunk_size=500):
    separators = ["\n\n", "\n", ".", "?", "!", " ", ""]
    # If text is short enough, return it as a single chunk
    if len(text) <= chunk_size:
        return [text]

    # Try splitting by progressively smaller units
    for sep in separators:
        split_list_of_text = split_by_separator(text, sep)
        chunks = []
        buffer = ""

        # Combine parts into chunks that fit the max_length
        for split_text in split_list_of_text:
            # Add next piece to buffer if it doesn't exceed max length
            if len(buffer) + len(split_text) <= chunk_size:
                buffer += split_text
            else:
                # Buffer full: save and start new buffer
                if buffer:
                    chunks.append(buffer.strip())
                buffer = split_text

        # Add any leftover buffer
        if buffer:
            chunks.append(buffer.strip())

        # If all resulting chunks are within limit, return them
        if all(len(c) <= chunk_size for c in chunks):
            return chunks

    return [text]

# Sliding Window Chunking
def sliding_window_chunking(text, nlp_pipeline='None', splitting_method='whitespace', chunk_size=200, step_size=100):
    # Split the text into tokens
    if splitting_method == 'whitespace':
        # Split the text into tokens based on whitespace
        tokens = text.split()
    else:
        # Split the text into tokens based on the selected method
        tokens = word_tokenization(text, nlp_pipeline, splitting_method)

    chunks = []

    # Slide a window through the tokens with the given step size
    for i in range(0, len(tokens), step_size):
        # Get a chunk of tokens from the current position
        chunk = tokens[i:i+chunk_size]
        # If the chunk has content, join into a string and store
        if chunk:
            chunks.append(' '.join(chunk))

    return chunks

# Topic Chunking
def topic_chunking(list_of_sentences, embeddings, clustering_method='kmeans', num_clusters=2, min_samples=2):
    labels = get_clutering_output(embeddings, clustering_method, num_clusters, min_samples)

    clustered_sentences = []
    current_cluster = labels[0]
    current_chunk = []

    for label, sentence in zip(labels, list_of_sentences):
        if label != current_cluster and current_chunk:
            clustered_sentences.append(" ".join(current_chunk))
            current_chunk = []
            current_cluster = label
        current_chunk.append(sentence)

    if current_chunk:
        clustered_sentences.append(" ".join(current_chunk))

    return clustered_sentences

# Semantic Chunking
def semantic_chunking(list_of_sentences, embeddings, sent_chunk_size=5):
    chunks = []
    current_chunk = []

    # Combine sentences into groups of chunk_size
    for i, emb in enumerate(embeddings):
        current_chunk.append(list_of_sentences[i])
        if len(current_chunk) >= sent_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    # Add any remaining sentences as the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# Hybrid Chunking
def hybrid_chunking(list_of_sentences, embeddings, sent_chunk_size=5, similarity_threshold=0.75):
    chunks = []
    current_chunk = [list_of_sentences[0]]
    for i in range(1, len(list_of_sentences)):
        # Compare similarity with the last sentence in the current chunk
        sim = util.pytorch_cos_sim(embeddings[i], embeddings[i-1]).item()
        current_chunk.append(list_of_sentences[i])

        # Start a new chunk if similarity drops or max size is hit
        if sim < similarity_threshold or len(current_chunk) >= sent_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks