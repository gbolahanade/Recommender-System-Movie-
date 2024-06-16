import streamlit as st
import pandas as pd
import pickle
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 25)
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.max_rows', 100)

Movies_df = pd.read_csv('Clean_processed_dataset.csv')
Movies_list = (Movies_df['Name'])
Movie_tags_similarity = pickle.load(open('Movie_tags_similarity.pkl', 'rb'))


def recommend(movie):
    movie_id = Movies_df[Movies_df['Name'] == movie]['Movie_id'].values[0]
    distances = Movie_tags_similarity[movie_id - 1]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])
    movie_list = movie_list[1:6]

    movies_recommended = []
    for i in movie_list:
        movies_recommended.append(Movies_df.iloc[i[0]].Name)
    return movies_recommended


st.write('Designed by TECHSQUAD  Team')
st.title('Movie Recommender System')

Movie_Selected = st.selectbox('Do you want a movie that captured your past interest? Make a selection below from more '
                              'than 5000 movies in the database.', Movies_list)

if st.button('Recommend'):
    recommendations = recommend(Movie_Selected)
    for x in recommendations:
        st.write(x)
