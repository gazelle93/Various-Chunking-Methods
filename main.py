import argparse
from utils import get_nlp_pipeline
from chunking_methods import fixed_size_chunking, recursive_chunking, sliding_window_chunking, semantic_chunking, hybrid_chunking

def main(args):
    selected_nlp_pipeline = 'None'
    if args.splitting_method == 'spacy':
        selected_nlp_pipeline = get_nlp_pipeline(args.splitting_method)

    if args.chunking_method == 'fixed_size':
        chunking_result = fixed_size_chunking(text=args.text, nlp_pipeline=selected_nlp_pipeline, splitting_method=args.word_splitting_method, chunk_size=args.chunk_size, overlap=args.overlap)
    elif args.chunking_method == 'recursive':
        chunking_result = recursive_chunking(text=args.text, chunk_size=args.chunk_size)
    elif args.chunking_method == 'sliding_window':
        chunking_result = sliding_window_chunking(text=args.text, nlp_pipeline=selected_nlp_pipeline, splitting_method=args.word_splitting_method, chunk_size=args.chunk_size, step_size=args.step_size)
    elif args.chunking_method == 'semantic':
        semantic_chunking(text, sent_transformer_model=args.sent_transformer_model, nlp_pipeline=selected_nlp_pipeline, splitting_method=args.sent_splitting_method, sent_chunk_size=args.chunk_size)
    elif args.chunking_method == 'hybrid':
        hybrid_chunking(text, sent_transformer_model=args.sent_transformer_model, nlp_pipeline=selected_nlp_pipeline, splitting_method=args.sent_splitting_method, sent_chunk_size=args.chunk_size, similarity_threshold=args.similarity_threshold)

    print(chunking_result)

if __name__ == "__main__":
    text = """
        Artificial intelligence has become an essential part of modern technology. From virtual assistants to self-driving cars, AI is integrated into daily life. These systems rely on large volumes of data and sophisticated algorithms to make decisions.

        One of the key components of many AI applications is natural language processing. NLP enables machines to understand and generate human language, making it possible for users to interact with systems in intuitive ways. Chatbots, document summarization tools, and voice assistants all depend on NLP.

        However, NLP systems often struggle with ambiguity and context. Words can have multiple meanings depending on how they are used. For example, the word “bank” can refer to a financial institution or the side of a river. Effective NLP models must consider the surrounding context to interpret meaning correctly.

        To address this, retrieval-augmented generation (RAG) techniques have been developed. RAG allows models to fetch relevant information from external sources before generating a response. This enhances both accuracy and factual grounding.

        Chunking plays a critical role in RAG systems. Documents must be divided into segments, or chunks, that are small enough for efficient retrieval but large enough to preserve context. Poorly chunked data can lead to irrelevant or confusing responses.

        Researchers have explored several chunking strategies. Fixed-size chunks are simple to implement but may break up coherent thoughts. Recursive chunking respects natural language structure, while semantic chunking uses embeddings to identify topical boundaries. A hybrid approach aims to combine the strengths of both.

        Selecting the right chunking method depends on the application. For simple lookup tasks, fixed-size chunks may be sufficient. For complex reasoning or customer support systems, semantic or hybrid chunking may yield better results.
        """

    parser = argparse.ArgumentParser()

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
            "(default: whitespace; options: whitespace, spacy, nltk, gensim)."
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
        "--chunking_method",
        default="fixed_size",
        type=str,
        help=(
            "Chunking method to apply "
            "(default: fixed_size; options: fixed_size, recursive, sliding_window, semantic, hybrid)."
        )
    )

    parser.add_argument(
        "--chunk_size",
        default=12,
        type=int,
        help="Maximum size of each chunk (number of characters, tokens, sentences, depending on context)."
    )

    parser.add_argument(
        "--overlap",
        default=0,
        type=int,
        help="Number of tokens or units to overlap between consecutive chunks."
    )

    parser.add_argument(
        "--step_size",
        default=0,
        type=int,
        help="Step size to move the sliding window for chunking (used in sliding_window method)."
    )

    parser.add_argument(
        "--sent_transformer_model",
        default="all-MiniLM-L6-v2",
        type=str,
        help="Name of the sentence transformer model used for semantic similarity."
    )

    parser.add_argument(
        "--similarity_threshold",
        default=0.75,
        type=float,
        help="Similarity threshold for merging chunks in semantic and hybrid chunking."
    )

    args = parser.parse_args()
    main(args)