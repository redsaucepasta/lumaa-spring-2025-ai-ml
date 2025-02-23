import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import gensim.downloader as api

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load stopwords
stop_words = set(stopwords.words('english'))

# Load a pretrained embedding model (GloVe, 50 dimensions)
embedding_model = api.load("glove-wiki-gigaword-50")

def preprocess_text(text):
    """
    Clean and tokenize text by lowercasing, removing punctuation, and removing stopwords.
    """
    # Convert to lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    return tokens

def get_column_vector(text, model):
    """
    Compute an average vector for a text column.
    Returns a zero vector if no tokens are found.
    """
    tokens = preprocess_text(text)
    vectors = [model[word] for word in tokens if word in model]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)

def get_weighted_movie_vector(row, weights, model):
    """
    Compute a weighted document vector for a movie from specific columns.
    
    Args:
        row (pd.Series): A row from the DataFrame.
        weights (dict): A dictionary mapping column names to their weights.
        model: Pretrained word embeddings model.
        
    Returns:
        A numpy array representing the weighted average vector.
    """
    weighted_vector = np.zeros(model.vector_size)
    total_weight = 0
    # For each column, compute the column vector and add weight * column_vector.
    for col, weight in weights.items():
        col_text = ' '.join(row[col]) if isinstance(row[col], list) else str(row[col])
        vec = get_column_vector(col_text, model)
        weighted_vector += weight * vec
        total_weight += weight
    if total_weight > 0:
        return weighted_vector / total_weight
    return weighted_vector

def clean_user_query(query):
    """
    Clean user query by removing common movie-search phrases and stopwords.
    Also handles special cases and normalizes text.
    """
    # Common phrases that don't add value to movie search
    remove_phrases = [
        'i want', 'i like', 'i love', 'show me', 'give me',
        'looking for', 'searching for', 'find me', 'recommend me',
        'movies with', 'movies about', 'films with', 'films about',
        'movies like', 'films like', 'similar to', 'something like',
        'can you recommend', 'please recommend', 'please show',
        'i am looking for', 'i am interested in'
    ]
    
    # Convert query to lowercase
    query = query.lower()
    
    # Remove the common phrases
    for phrase in remove_phrases:
        query = query.replace(phrase, '')
    
    # Remove extra whitespace and strip
    query = ' '.join(query.split())
    
    # Apply standard preprocessing (stopwords removal, etc.)
    processed_tokens = preprocess_text(query)
    
    # Join tokens back together
    cleaned_query = ' '.join(processed_tokens)
    
    # Handle empty queries
    if not cleaned_query.strip():
        return query  # Return original query if cleaning removed everything
        
    return cleaned_query

def recommend_movies_weighted(user_query, df, weights, top_n=5, reg=True):
    """
    Recommend movies based on a user's query using a weighted system over specific columns.
    
    Args:
        user_query (str): User input description.
        df (pd.DataFrame): DataFrame containing movie data.
        weights (dict): Column weights (e.g., {'overview': 2, 'keywords': 3, 'genres': 1, ...}).
        top_n (int): Number of recommendations.
        reg (bool): If True, applies L2 normalization to the vectors (regularization).
        
    Returns:
        DataFrame of top N movies with similarity scores.
    """
    # Clean and preprocess user query
    user_query = clean_user_query(user_query)
    # print(f"Processed query: {user_query}")

    # Precompute weighted document vectors for each movie
    doc_vectors = np.array([
        get_weighted_movie_vector(row, weights, embedding_model) 
        for _, row in df.iterrows()
    ])
    
    # Compute query vector (simple average)
    query_vector = get_column_vector(user_query, embedding_model).reshape(1, -1)
    
    # Apply L2 normalization to regularize vector magnitudes
    if reg:
        doc_vectors = normalize(doc_vectors, norm='l2')
        query_vector = normalize(query_vector, norm='l2')
    
    # Compute cosine similarity
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    recommendations = df.iloc[top_indices].copy()
    recommendations['similarity_score'] = similarities[top_indices]
    return recommendations[['title', 'genres', 'cast', 'similarity_score']]

if __name__ == "__main__":
    # Example weights (adjust based on your domain)
    column_weights = {
        'overview': 2.5,
        'genres': 1.5,
        'keywords': 2.0,
        'tagline': 1.0,
        'cast': 1.5,
        'director': 1.0,
        'title_list': 2.0
    }
    
    user_query = input("\nEnter the prompt: ")

    movies_df = pd.read_csv('movies_cleaned_preprocessed.csv')
    
    recommendations = recommend_movies_weighted(user_query, movies_df, column_weights, top_n=5, reg=True)
    
    print("\nWeighted Recommendations with Regularization:")
    print(recommendations.head())
