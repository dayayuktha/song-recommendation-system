import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set page title
st.set_page_config(page_title="Song Recommender", layout="wide")
st.title("🎵 Song Recommendation System by Yuktha Dayanand")

# Load and prepare the data
@st.cache_data
def load_data():
    df = pd.read_csv(r'tcc_ceds_music.csv')
    return df

def prepare_features(df):
    # Select relevant features for similarity calculation
    feature_cols = [
        'danceability', 'loudness', 'acousticness', 
        'instrumentalness', 'valence', 'energy'
    ]
    
    # Create genre dummy variables
    genre_dummies = pd.get_dummies(df['genre'], prefix='genre')
    
    # Create topic dummy variables
    topic_dummies = pd.get_dummies(df['topic'], prefix='topic')
    
    # Combine all features
    features_df = pd.concat([
        df[feature_cols],
        genre_dummies,
        topic_dummies
    ], axis=1)
    
    # Scale the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)
    
    return features_scaled, features_df.columns

def get_recommendations(df, features_scaled, song_idx, n_recommendations=5):
    # Calculate cosine similarity
    similarity = cosine_similarity([features_scaled[song_idx]], features_scaled)
    
    # Get indices of most similar songs
    similar_indices = similarity[0].argsort()[::-1][1:n_recommendations+1]
    
    # Get the recommended songs
    recommendations = df.iloc[similar_indices][['artist_name', 'track_name', 'genre', 'topic']]
    
    return recommendations

# Load the data
try:
    df = load_data()
    features_scaled, feature_names = prepare_features(df)
    
    # Create the search interface
    st.subheader("Select a Song")
    
    # Add artist filter
    artists = sorted(df['artist_name'].unique())
    selected_artist = st.selectbox("Filter by Artist", ['All Artists'] + list(artists))
    
    # Filter songs by selected artist
    if selected_artist != 'All Artists':
        song_options = df[df['artist_name'] == selected_artist]
    else:
        song_options = df
        
    # Create song selection dropdown
    selected_song = st.selectbox(
        "Choose a Song",
        song_options.apply(lambda x: f"{x['artist_name']} - {x['track_name']}", axis=1)
    )
    
    if st.button("Get Recommendations"):
        # Get the index of the selected song
        artist_name, track_name = selected_song.split(" - ", 1)
        song_idx = df[
            (df['artist_name'] == artist_name) & 
            (df['track_name'] == track_name)
        ].index[0]
        
        # Get recommendations
        recommendations = get_recommendations(df, features_scaled, song_idx)
        
        # Display the original song's details
        st.subheader("Selected Song Details")
        original_song = df.iloc[song_idx]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Genre", original_song['genre'])
        with col2:
            st.metric("Topic", original_song['topic'])
        with col3:
            st.metric("Danceability", f"{original_song['danceability']:.2f}")
        with col4:
            st.metric("Energy", f"{original_song['energy']:.2f}")
        
        # Display recommendations
        st.subheader("Recommended Songs")
        for i, (_, row) in enumerate(recommendations.iterrows(), 1):
            st.markdown(f"""
            **{i}. {row['artist_name']} - {row['track_name']}**  
            Genre: {row['genre']} | Topic: {row['topic']}
            """)
            
except Exception as e:
    st.error(f"Error: Make sure the file 'ds.csv' is in the same directory as this script and contains the required columns. Error details: {str(e)}")

# Add footer with explanation
st.markdown("---")
st.markdown("""
**How it works:**
- The recommendation system uses features like audio characteristics (danceability, loudness, etc.), genre, and topic
- It finds similar songs using cosine similarity between these features
- Songs are recommended based on their overall similarity to your selected song
""")