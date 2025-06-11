# Overview
This repository explores various chunking strategies for improving the efficiency and effectiveness of Retrieval-Augmented Generation (RAG) pipelines. Chunking determines how source documents are segmented before being embedded and retrieved, which can significantly affect retrieval quality and latency.

## Motivation

Chunking plays a critical role in balancing context preservation, retrieval precision, and inference cost. This project compares common and advanced methods under a controlled evaluation framework.

## Repository Structure

- `chunking_mehtods.py`: Contains implementations of chunking strategies such as:
  - Fixed-size chunking
  - Recursive chunking
  - Sliding chunking
  - Semantic chunking
  - Hybrid chunking
- `utils.py`: Utility functions shared across modules.

## Methods Compared

| Chunking Method      | Strategy                             | Pros                            | Cons                             |
|----------------------|--------------------------------------|----------------------------------|----------------------------------|
| Fixed-size           | Uniform length split                 | Simple, fast                    | Can break semantic units         |
| Recursive            | Uses hierarchical splitting rules    | Maintains structure             | Slower, heuristic-based          |
| Sliding window       | Overlapping segments                 | High recall                     | Increases redundancy             |
| Semantic             | Embedding-based or topic-aware       | Semantic coherence              | More complex to implement        |
| Hybrid             | Text-structure + semantic similarity       | Balanced, readable and coherent | More complex logic and slower    |


## Prerequisites
spacy
nltk
gensim
sentence-transformers
numpy
