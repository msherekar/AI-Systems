import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from preprocess import normalization


def encode(data, method_encode, embedding_dim=100, window=5, min_count=1):
    if method_encode == "bag_of_words":
        vectorizer = CountVectorizer()
        encoded_data = vectorizer.fit_transform([data])  # Wrap the string in a list
        return encoded_data.toarray()[0]  # Return the first element of the array

    elif method_encode == "tfidf":
        vectorizer = TfidfVectorizer()
        encoded_data = vectorizer.fit_transform([data])
        feature_names = vectorizer.get_feature_names_out()
        return encoded_data.toarray()[0], feature_names

    elif method_encode == "word2vec":
        # Tokenize the data
        tokenized_data = [word_tokenize(data.lower())]

        # Train Word2Vec model
        model = Word2Vec(sentences=tokenized_data,
                         vector_size=embedding_dim,
                         window=window,
                         min_count=min_count)

        # Get Word2Vec embeddings for each word
        embeddings = {word: model.wv[word] for word in model.wv.index_to_key}

        return embeddings

    else:
        raise ValueError("Invalid encoding method. Choose 'bag_of_words', 'tfidf', or 'word2vec'.")

# def encode(data, method="bag_of_words", embedding_dim=100, window=5, min_count=1):
#     if method == "bag_of_words":
#
#         vectorizer = CountVectorizer()
#         encoded_data = vectorizer.fit_transform(data)
#         return encoded_data.toarray()
#
#     elif method == "tfidf":
#         # TF-IDF encoding
#         vectorizer = TfidfVectorizer()
#         encoded_data = vectorizer.fit_transform(data)
#         feature_names = vectorizer.get_feature_names_out()
#         return encoded_data.toarray(), feature_names
#
#     elif method == "word2vec":
#         # Tokenize the data
#         tokenized_data = [word_tokenize(sentence.lower()) for sentence in data]
#
#         # Train Word2Vec model
#         model = Word2Vec(sentences=tokenized_data,
#                          vector_size=embedding_dim,
#                          window=window,
#                          min_count=min_count)
#
#         # Get Word2Vec embeddings for each word
#         embeddings = {word: model.wv[word] for word in model.wv.index_to_key}
#
#         return embeddings
#
#     else:
#         raise ValueError("Invalid encoding method. Choose 'bag_of_words', 'tfidf', or 'word2vec'.")


# if __name__ == "__main__":
#     data = ["I can't believe it's already 2021. I'm so excited for the new year.",
#             "I like apples. I also like bananas.",
#             "I like apples and bananas. I also like grapes."]
#
#     preprocessed_data = [normalization(sentence) for sentence in data]
#     print(preprocessed_data)
#
#     # Bag-of-Words encoding
#     # encoded_data, feature_names = encode(preprocessed_data, method="bag_of_words")
#     # print("Bag-of-Words Encoding:")
#     # print(encoded_data)
#     # print("Feature Names:")
#     # print(feature_names)
#
#     # encoded_data, feature_names = encode(preprocessed_data, method="tfidf")
#     # print("\nTF-IDF Encoding:")
#     # print(encoded_data)
#     # print("Feature Names:")
#     # print(feature_names)
#
#     # Word2Vec encoding
#     word2vec_embeddings = encode(preprocessed_data, method="word2vec")
#     print("\nWord2Vec Embeddings:")
#     for word, embedding in word2vec_embeddings.items():
#         print(f"Word: {word}, Embedding: {embedding}")
