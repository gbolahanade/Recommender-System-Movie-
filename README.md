# Movie Recommender System

## Overview
This repository contains the code and resources for a Movie Recommender System. The project is divided into three main parts:

    IMDB Web Data Scraping
    Data Preparation and Model Implementation
    Web Deployment Using Streamlit

## Table of Contents
1. Project Task
2. IMDB Web Data Scraping
3. Data Preparation and Model Implementation
4. Web Deployment Using Streamlit
5. How to Run the Project
6. Authors

## Project Task
In this assessment, students will work in groups of 3 or 4. In this module, we
have been discussing several applications of artificial intelligence, machine
learning and data science. The following is a list of some selected topics from
the course:
1. Recommender Systems
2. Reinforcement learning
1. What is the task for this assessment?
3. Adversarial Learning
4. Image or face recognition
5. Social network analysis
6. Speech/Voice recognition
7. Any other application of your in the field of AI, ML and DS that is not listed above
You can choose any one of the above topics that interest you. Based on the
topic that you have selected, you have to write a group report and implement
code.

## IMDB Web Data Scraping
### Overview

This part of the project involves scraping movie data from IMDB. The ImdbMovieScrapper class is designed to extract movie details such as title, release date, runtime, genre, rating, score, description, director, stars, votes, and gross earnings.

```
import lxml
import re
import numpy as np
import requests
from tabulate import tabulate
import pandas as pd
from bs4 import BeautifulSoup

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 25)
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.max_rows', 100)

class ImdbMovieScrapper:
    # Class variables for storing movie data
    MovieTitleList = []
    MovieDateList = []
    MovieRunTimeList = []
    MovieGenreList = []
    MovieRatingList = []
    MovieScoreList = []
    MovieDescriptionList = []
    MovieDirectorList = []
    MovieStarsList = []
    MovieVotesList = []
    MovieGrossList = []

    def __init__(self, url):
        self.PageResult = requests.get(url)
        self.PageContent = self.PageResult.content
        self.soup = BeautifulSoup(self.PageContent, features="lxml")

    def MovieDivBody(self):
        return self.soup.find_all("div", {'class': 'lister-item mode-advanced'})

    def MovieData(self):
        for EachMovieSection in self.MovieDivBody():
            movie_title_header = EachMovieSection.find("h3", class_="lister-item-header")
            movie_title_anchor = movie_title_header.find("a").text
            self.MovieTitleList.append(movie_title_anchor)

            movie_date_tag = movie_title_header.find("span", class_="lister-item-year").text.strip("()")
            self.MovieDateList.append(movie_date_tag)

            movie_runtime_tag = EachMovieSection.find("span", class_="runtime")
            self.MovieRunTimeList.append(movie_runtime_tag.text[:-4] if movie_runtime_tag else np.nan)

            movie_genre_tag = EachMovieSection.find("span", class_="genre")
            self.MovieGenreList.append(movie_genre_tag.text.rstrip().replace("\n", "").split(",") if movie_genre_tag else np.nan)

            movie_rating_tag = EachMovieSection.find("strong")
            self.MovieRatingList.append(movie_rating_tag.text if movie_rating_tag else np.nan)

            movie_score_tag = EachMovieSection.find("span", class_="metascore mixed")
            self.MovieScoreList.append(movie_score_tag.text.rstrip() if movie_score_tag else np.nan)

            movie_description_tag = EachMovieSection.find_all("p", class_="text-muted")
            self.MovieDescriptionList.append(movie_description_tag[-1].text.lstrip())

            movie_director_tag = EachMovieSection.find("p", class_="")
            movie_director_cast = movie_director_tag.text.replace("\n", "").split('|') if movie_director_tag else []
            movie_director_cast = [x.strip() for x in movie_director_cast]
            movie_director_cast = [movie_director_cast[i].replace(j, "") for i, j in enumerate(["Director:", "Stars:"])]
            self.MovieDirectorList.append(movie_director_cast[0] if len(movie_director_cast) > 0 else np.nan)
            self.MovieStarsList.append([x.strip() for x in movie_director_cast[1].split(",")] if len(movie_director_cast) > 1 else np.nan)

            movie_votes_gross_tag = EachMovieSection.find_all("span", attrs={"name": "nv"})
            self.MovieVotesList.append(movie_votes_gross_tag[0].text if len(movie_votes_gross_tag) > 0 else np.nan)
            self.MovieGrossList.append(movie_votes_gross_tag[1].text if len(movie_votes_gross_tag) > 1 else np.nan)

        movie_data = [self.MovieTitleList, self.MovieDateList, self.MovieRunTimeList, self.MovieGenreList, self.MovieRatingList, self.MovieScoreList, self.MovieDescriptionList, self.MovieDirectorList, self.MovieStarsList, self.MovieVotesList, self.MovieGrossList]
        return movie_data

    def MovieDataFrame(self):
        movie_data = self.MovieData()
        movie_dataframe = pd.DataFrame({'MovieTitle': movie_data[0], 'MovieDate': movie_data[1], 'MovieRunTime': movie_data[2], 'MovieGenre': movie_data[3], 'MovieRating': movie_data[4], 'MovieScore': movie_data[5], 'MovieDescription': movie_data[6], 'MovieDirector': movie_data[7], 'MovieStars': movie_data[8], 'MovieVotes': movie_data[9], 'MovieGross': movie_data[10]})
        return movie_dataframe

# Scraping data from 51 IMDB pages
data_frame_list = []
for number in range(51):
    url = f"https://www.imdb.com/search/title?count=100&title_type=feature,tv_series&ref_=nv_wl_img_2&start={number * 100 + 1}"
    page = ImdbMovieScrapper(url)
    dataframe = page.MovieDataFrame()
    data_frame_list.append(dataframe)

data_frame_list = pd.concat(data_frame_list)
print(tabulate(data_frame_list, headers='keys', tablefmt='fancy_grid'))
```

### Output
The scraped data is saved to a CSV file named Movie_Final_Dataframe.csv.


## Data Preparation and Model Implementation
### Overview
This section involves cleaning the scraped data, engineering features, and building a content-based recommender system using TF-IDF vectorization and cosine similarity.


```
import pandas as pd 
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option('display.max_colwidth', 200)
pd.set_option('display.max_columns', 100)

# Loading the dataset
df = pd.read_csv('Movie_Final_Dataframe.csv')

# Data Cleaning and Feature Engineering
df['Movie_id'] = [i for i in range(1, df.shape[0]+1)]
df['Name'] = df['MovieTitle']
df['Year'] = df['MovieDate'].str.replace('[IV)(]', '', regex=True)
df['Time'] = df['MovieRunTime']
df['Genre'] = df['MovieGenre'].str.replace("""[""'\[\],]""", '', regex=True)
df['Rating'] = df['MovieRating']
df['Score'] = df['MovieScore']
df['Description'] = df['MovieDescription'].str.replace("""[""'\[\],]""", '', regex=True)
df['Directors_cast'] = df['MovieDirector'].str.replace("(Directors:)", '', regex=True)
df['Stars'] = df['MovieStars'].str.replace("""[""\['\]]""", '', regex=True).str.replace("(^Stars:)", '', regex=True)
df['Votes'] = df['MovieVotes']
df['Total'] = df['MovieGross']

df.drop(["Unnamed: 0", "MovieTitle", "MovieDate", "MovieRunTime", "MovieGenre", "MovieRating", "MovieScore", "MovieDescription", "MovieDirector", "MovieStars", "MovieVotes", "MovieGross"], axis=1, inplace=True)

# Feature Engineering
df['Tags'] = df['Genre'].fillna('') + ' ' + df['Description'].fillna('') + ' ' + df['Directors_cast'].str.replace(' ', '').fillna('') + ' ' + df['Stars'].str.replace(' ', '').str.replace(',', ' ').fillna('')
df['Tags'] = df['Tags'].apply(lambda x: x.lower())

# Text Preprocessing
import string
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

df['Tags'] = df['Tags'].apply(remove_punctuation)

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stopwords.words('english')])

df['Tags'] = df['Tags'].apply(remove_stopwords)

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

df['Tags'] = df['Tags'].apply(stem)

# Vectorization and Similarity Calculation
vectorizer = TfidfVectorizer()
Movie_tags_vec = vectorizer.fit_transform(df["Tags"]).toarray()
Movie_tags_similarity = cosine_similarity(Movie_tags_vec)

# Save Similarity Matrix
import pickle
pickle.dump(Movie_tags_similarity, open('Movie_tags_similarity.pkl', 'wb'))

# Recommendation Function
def recommend(movie):
    Movie_id = df[df['Name'] == movie]['Movie_id'].values[0]
    Similarity_score = list(enumerate(Movie_tags_similarity[Movie_id-1]))
    Similarity_score = sorted(Similarity_score, key=lambda x: x[1], reverse=True)
    Similarity_score = Similarity_score[1:6]
    Movie_Index = [i[0] for i in Similarity_score]
    return df.iloc[Movie_Index][['Name', 'Year', 'Genre', 'Rating', 'Score', 'Description', 'Directors_cast', 'Stars']]

# Example Recommendation
print(recommend("Batman"))
```
### Output
The preprocessed data is saved to Movie_Final_Dataframe.csv, and the similarity matrix is saved as Movie_tags_similarity.pkl.


## Web Deployment Using Streamlit
### Overview
This part of the project involves creating a web application using Streamlit to allow users to interact with the movie recommender system.

```
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# Load pre-trained model and data
df = pd.read_csv('Movie_Final_Dataframe.csv')
Movie_tags_similarity = pickle.load(open('Movie_tags_similarity.pkl', 'rb'))

def recommend(movie):
    Movie_id = df[df['Name'] == movie]['Movie_id'].values[0]
    Similarity_score = list(enumerate(Movie_tags_similarity[Movie_id-1]))
    Similarity_score = sorted(Similarity_score, key=lambda x: x[1], reverse=True)
    Similarity_score = Similarity_score[1:6]
    Movie_Index = [i[0] for i in Similarity_score]
    return df.iloc[Movie_Index][['Name', 'Year', 'Genre', 'Rating', 'Score', 'Description', 'Directors_cast', 'Stars']]

# Streamlit Web App
st.title('Movie Recommender System')
movie_list = df['Name'].values
selected_movie = st.selectbox('Type or select a movie from the dropdown', movie_list)

if st.button('Show Recommendation'):
    recommended_movie_names = recommend(selected_movie)
    for i in range(len(recommended_movie_names)):
        st.subheader(recommended_movie_names.iloc[i][0])
        st.write('Year:', recommended_movie_names.iloc[i][1])
        st.write('Genre:', recommended_movie_names.iloc[i][2])
        st.write('Rating:', recommended_movie_names.iloc[i][3])
        st.write('Score:', recommended_movie_names.iloc[i][4])
        st.write('Description:', recommended_movie_names.iloc[i][5])
        st.write('Director:', recommended_movie_names.iloc[i][6])
        st.write('Stars:', recommended_movie_names.iloc[i][7])
```

## How to Run the Project

    IMDB Web Data Scraping: Run the Python script to scrape the movie data and save it to a CSV file.
    Data Preparation and Model Implementation: Run the Python script to clean the data, engineer features, and build the content-based recommender system. Save the processed data and similarity matrix.
    Web Deployment Using Streamlit: Run the Streamlit app to deploy the movie recommender system on the web.

## Authors
TechSquad Team (Gbolahan Michael Adebesin, Folasayo, and others)