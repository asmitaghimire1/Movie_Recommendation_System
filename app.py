import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Loading Dataset

movies_data = pd.read_csv('movies.csv')

# Filling missing values for selected features
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combining features
combined_features = (
    movies_data['genres'] + ' ' +
    movies_data['keywords'] + ' ' +
    movies_data['tagline'] + ' ' +
    movies_data['cast'] + ' ' +
    movies_data['director']
)

# Converting text to TF-IDF vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Compute cosine similarity
similarity = cosine_similarity(feature_vectors)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎬 Movie Recommendation System")

# Dropdown for movie selection
movie_name = st.text_input("Enter your favourite movie:")

if st.button("Recommend"):
    try:
        # Find closest match (optional because using dropdown)
        find_close_match = difflib.get_close_matches(movie_name, movies_data['title'].tolist())
        close_match = find_close_match[0]

        # Get index of the movie
        index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

        # Get similarity scores
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        # Display top 10 recommended movies
        st.subheader("Movies suggested for you:")
        for i, movie in enumerate(sorted_similar_movies[1:11]):  # skip first (same movie)
            index = movie[0]
            title_from_index = movies_data[movies_data.index == index]['title'].values[0]
            st.write(f"{i+1}. {title_from_index}")

    except:
        st.error("Movie not found! Please check your selection.")

