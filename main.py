import argparse
from utils import get_nlp_pipeline, text_splitting_and_embedding
from chunking_methods import fixed_size_chunking, recursive_chunking, sliding_window_chunking, semantic_chunking, \
    hybrid_chunking, topic_chunking


def get_chunking_result(args):
    if args.word_splitting_method == 'spacy' or args.sent_splitting_method == 'spacy':
        selected_nlp_pipeline = get_nlp_pipeline('spacy')

    if args.chunking_method == 'fixed_size':
        return fixed_size_chunking(text=args.text, nlp_pipeline=selected_nlp_pipeline,
                                   splitting_method=args.word_splitting_method,
                                   hunk_size=args.chunk_size, overlap=args.overlap)
    elif args.chunking_method == 'recursive':
        return recursive_chunking(text=args.text, chunk_size=args.chunk_size)
    elif args.chunking_method == 'sliding_window':
        return sliding_window_chunking(text=args.text, nlp_pipeline=selected_nlp_pipeline,
                                       splitting_method=args.word_splitting_method,
                                       chunk_size=args.chunk_size, step_size=args.step_size)
    else:
        if args.chunking_method in ['semantic', 'hybrid', 'topic']:
            list_of_sentences, embeddings = text_splitting_and_embedding(text=args.text,
                                                                         nlp_pipeline=selected_nlp_pipeline,
                                                                         splitting_method=args.sent_splitting_method,
                                                                         embedding_method=args.embedding_method,
                                                                         embedding_model=args.embedding_model)

            if args.chunking_method == 'topic':
                return topic_chunking(list_of_sentences=list_of_sentences, embeddings=embeddings,
                                      clustering_method=args.clustering_method,
                                      num_clusters=args.num_clusters, min_samples=args.min_samples)
            elif args.chunking_method == 'semantic':
                return semantic_chunking(list_of_sentences=list_of_sentences, embeddings=embeddings,
                                         sent_chunk_size=args.chunk_size)
            elif args.chunking_method == 'hybrid':
                return hybrid_chunking(list_of_sentences=list_of_sentences, embeddings=embeddings,
                                       sent_chunk_size=args.chunk_size, similarity_threshold=args.similarity_threshold)
    return None


def main(args):
    chunking_result = get_chunking_result(args)
    if chunking_result == None:
        print("Please select one of the chunking method options.")
    else:
        print("Selected Chunking Method: {}".format(args.chunking_method))
        print("Chunking Results")
        for idx, chunk in enumerate(chunking_result):
            print("Chunk {}: {}".format(idx, chunk))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text",
        default="""
        Artificial intelligence has become an essential part of modern technology. From virtual assistants to self-driving cars, AI is integrated into daily life. These systems rely on large volumes of data and sophisticated algorithms to make decisions.
        One of the key components of many AI applications is natural language processing. NLP enables machines to understand and generate human language, making it possible for users to interact with systems in intuitive ways. Chatbots, document summarization tools, and voice assistants all depend on NLP.
        However, NLP systems often struggle with ambiguity and context. Words can have multiple meanings depending on how they are used. For example, the word “bank” can refer to a financial institution or the side of a river. Effective NLP models must consider the surrounding context to interpret meaning correctly.
        To address this, retrieval-augmented generation (RAG) techniques have been developed. RAG allows models to fetch relevant information from external sources before generating a response. This enhances both accuracy and factual grounding.
        Chunking plays a critical role in RAG systems. Documents must be divided into segments, or chunks, that are small enough for efficient retrieval but large enough to preserve context. Poorly chunked data can lead to irrelevant or confusing responses.
        Researchers have explored several chunking strategies. Fixed-size chunks are simple to implement but may break up coherent thoughts. Recursive chunking respects natural language structure, while semantic chunking uses embeddings to identify topical boundaries. A hybrid approach aims to combine the strengths of both.
        Selecting the right chunking method depends on the application. For simple lookup tasks, fixed-size chunks may be sufficient. For complex reasoning or customer support systems, semantic or hybrid chunking may yield better results.

        Bookkeeping records income and expenses. Financial statements summarize performance.
        Balance sheets show assets and liabilities. Income statements report profits and losses.
        Auditors verify accuracy of reports. Tax accounting ensures compliance with laws.
        Technology automates routine accounting tasks. Cloud platforms store financial data securely.
        Despite automation, accountants must ensure data integrity. Ethical standards guide financial reporting.

        AI is transforming industries. Machine learning, a subset of AI, allows computers to learn from data.
        NLP enables machines to understand human language. Many businesses use AI for customer service.
        Chatbots handle routine inquiries. In healthcare, AI diagnoses diseases. Radiology uses AI to interpret images.
        Despite benefits, AI raises ethical concerns. Data privacy and bias are major issues. Governments draft regulations.
        Accounting tracks financial transactions. It ensures businesses follow legal standards.
        """,
        type=str,
        help="Input text for chunking experiment."
    )

    parser.add_argument(
        "--chunking_method",
        default="fixed_size",
        type=str,
        help=(
            "Chunking method to apply "
            "(default: fixed_size; options: fixed_size, recursive, sliding_window, topic, semantic, hybrid)."
        )
    )

    parser.add_argument(
        "--nlp_pipeline",
        default="spacy",
        type=str,
        help="NLP preprocessing pipeline to use (default: spacy)."
    )

    parser.add_argument(
        "--word_splitting_method",
        default="whitespace",
        type=str,
        help=(
            "Word splitting method for fixed-size and recursive chunking "
            "(default: whitespace; options: whitespace, spacy, nltk)."
        )
    )

    parser.add_argument(
        "--sent_splitting_method",
        default="regex",
        type=str,
        help=(
            "Sentence splitting method for semantic and hybrid chunking "
            "(default: regex; options: spacy, nltk, regex)."
        )
    )

    parser.add_argument(
        "--chunk_size",
        default=100,
        type=int,
        help=(
            "Maximum size of each chunk; number of characters, tokens, sentences, depending on context"
            "(default: 100)."
        )
    )

    parser.add_argument(
        "--overlap",
        default=0,
        type=int,
        help="Number of tokens or units to overlap between consecutive chunks."
    )

    parser.add_argument(
        "--step_size",
        default=100,
        type=int,
        help="Step size to move the sliding window for chunking (used in sliding_window method)."
    )

    parser.add_argument(
        "--embedding_method",
        default="tfidf",
        type=str,
        help=(
            "The name of the embedding methods"
            "(default: tfidf; options: tfidf, sentence_embedding)."
        )
    )

    parser.add_argument(
        "--embedding_model",
        default="all-MiniLM-L6-v2",
        type=str,
        help=(
            "The name of the sentence transformer model"
            "(default: all-MiniLM-L6-v2)."
        )
    )

    parser.add_argument(
        "--similarity_threshold",
        default=0.75,
        type=float,
        help="Similarity threshold for merging chunks in semantic and hybrid chunking."
    )

    parser.add_argument(
        "--clustering_method",
        default='kmeans',
        type=str,
        help=(
            "The name of the clustering method"
            "(default: kmeans; options: kmeans, dbscan)."
        )
    )

    parser.add_argument(
        "--num_clusters",
        default=2,
        type=int,
        help="The number of clusters to form as well as the number of centroids to generate.."
    )

    parser.add_argument(
        "--min_samples",
        default=2,
        type=int,
        help="The number of samples (or total weight) in a neighborhood for a point to be considered as a core point."
    )
    args = parser.parse_args()
    main(args)