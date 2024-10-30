# song-recommendation-system
This project is a Song Recommendation System built using Streamlit that suggests similar songs based on user input. The system leverages audio features and metadata from a music dataset (such as genre, topic, and characteristics like danceability and energy). Here’s a brief overview of how it works:

Data Loading and Preprocessing:

The data is loaded from a CSV file containing song information.
Important audio features (e.g., danceability, loudness, acousticness) are selected, and categorical data like genres and topics are converted into numerical dummy variables.
The features are standardized to ensure comparability.
Cosine Similarity for Recommendations:

Cosine similarity is used to calculate the similarity between songs based on the chosen features.
When a user selects a song, the app finds and recommends the most similar songs in the dataset by calculating cosine similarity scores.
User Interaction:

The user can filter songs by artist, select a specific song, and get a list of similar song recommendations.
The app displays both the details of the selected song and a list of recommended songs with relevant metadata.
This project showcases data preprocessing, cosine similarity for recommendations, and an interactive UI with Streamlit.






