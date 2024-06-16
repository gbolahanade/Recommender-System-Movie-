import lxml
import re
import numpy as np
import requests
from tabulate import tabulate
import pandas as pd

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 25)
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.max_rows', 100)
from bs4 import BeautifulSoup
from requests import get


class ImdbMovieScrapper:
    # Class variable
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

    # movie_data = []

    # constructor
    def __init__(self, url):
        # Request content from a web page
        # self.movie_data = None
        self.PageResult = requests.get(url)
        self.PageContent = self.PageResult.content

        # Set as Beautiful Soup Object
        self.soup = BeautifulSoup(self.PageContent, features="lxml")

    # Instance method
    def MovieDivBody(self):
        # Go to the Page_content with sections/divs of interest i.e. Movie sections
        movie_div_body = self.soup.find_all("div", {'class': 'lister-item mode-advanced'})
        return movie_div_body

    # Instance method
    def MovieData(self):
        for EachMovieSection in self.MovieDivBody():
            # MovieTitleList Update
            movie_title_header = EachMovieSection.find("h3", class_="lister-item-header")
            movie_title_anchor = movie_title_header.find("a").text
            self.MovieTitleList.append(movie_title_anchor)

            # MovieDateList Update
            movie_date_tag = movie_title_header.find("span", class_="lister-item-year").text.strip("()")
            self.MovieDateList.append(movie_date_tag)

            # MovieRunTimeList Update
            movie_runtime_tag = EachMovieSection.find("span", class_="runtime")
            try:
                self.MovieRunTimeList.append(
                    movie_runtime_tag.text[:-4])  # stripping off the min amd space portion of the runtime
            except:
                self.MovieRunTimeList.append(np.nan)

            # MovieGenreList  Update
            movie_genre_tag = EachMovieSection.find("span", class_="genre")
            try:
                self.MovieGenreList.append(movie_genre_tag.text.rstrip().replace("\n", "").split(","))
            except:
                self.MovieGenreList.append(np.nan)

            # MovieRatingList Update
            movie_rating_tag = EachMovieSection.find("strong")
            try:
                self.MovieRatingList.append(movie_rating_tag.text)
            except:
                self.MovieRatingList.append(np.nan)

            # MovieScoreList Update
            movie_score_tag = EachMovieSection.find("span", class_="metascore mixed")
            # Note that some Movies do not have their rating recorded, hence will replace that with null value (nan)
            try:
                self.MovieScoreList.append(movie_score_tag.text.rstrip())
            except:
                self.MovieScoreList.append(np.nan)

            # MovieDescriptionList
            movie_description_tag = EachMovieSection.find_all("p", class_="text-muted")
            self.MovieDescriptionList.append(movie_description_tag[-1].text.lstrip())

            # MovieDirectorList and MovieStarList update
            movie_director_tag = EachMovieSection.find("p", class_="")
            try:
                movie_director_cast = movie_director_tag.text.replace("\n", "").split('|')
                movie_director_cast = [x.strip() for x in movie_director_cast]
                movie_director_cast = [movie_director_cast[i].replace(j, "") for i, j in
                                       enumerate(["Director:", "Stars:"])]
                self.MovieDirectorList.append(movie_director_cast[0])
                self.MovieStarsList.append([x.strip() for x in movie_director_cast[1].split(",")])
            except:
                movie_director_cast = movie_director_tag.text.replace("\n", "").strip()
                self.MovieDirectorList.append(np.nan)
                self.MovieStarsList.append([x.strip() for x in movie_director_cast.split(",")])

            # MovieVotesList and MovieGrossList Update
            movie_votes_gross_tag = EachMovieSection.find_all("span", attrs={"name": "nv"})
            if len(movie_votes_gross_tag) == 2:
                self.MovieVotesList.append(movie_votes_gross_tag[0].text)
                self.MovieGrossList.append(movie_votes_gross_tag[1].text)
            elif len(movie_votes_gross_tag) == 1:
                self.MovieVotesList.append(movie_votes_gross_tag[0].text)
                self.MovieGrossList.append(np.nan)
            else:
                self.MovieVotesList.append(np.nan)
                self.MovieGrossList.append(np.nan)

        movie_data = [self.MovieTitleList, self.MovieDateList, self.MovieRunTimeList, self.MovieGenreList,
                      self.MovieRatingList, self.MovieScoreList, self.MovieDescriptionList,
                      self.MovieDirectorList, self.MovieStarsList, self.MovieVotesList,
                      self.MovieGrossList, ]

        return movie_data

    # Instance method
    def MovieDataFrame(self):
        movie_data = self.MovieData()
        movie_dataframe = pd.DataFrame({'MovieTitle': movie_data[0], 'MovieDate': movie_data[1],
                                        'MovieRunTime': movie_data[2], 'MovieGenre': movie_data[3],
                                        'MovieRating': movie_data[4], 'MovieScore': movie_data[5],
                                        'MovieDescription': movie_data[6], 'MovieDirector': movie_data[7],
                                        'MovieStars': movie_data[8], 'MovieVotes': movie_data[9],
                                        'MovieGross': movie_data[10]})
        return movie_dataframe


# Calling the IMDB MovieScrapper class to scrape 51 Imdb pages for 5100 datapoints

# Initializing a list container to store the movies scrapped data
data_frame_list = []
dataframe = None

# looping through the 51 Imdb movie webpages
for number in range(51):
    page_count = number
    if page_count == 0:
        start_url = "https://www.imdb.com/search/title?count=100&title_type=feature,tv_series&ref_=nv_wl_img_2"
        page = ImdbMovieScrapper(start_url)
        dataframe = page.MovieDataFrame()
    elif page_count > 0 < 51:
        next_url = "https://www.imdb.com/search/title/?title_type=feature,tv_series&count=100&start=" \
                   + str(number) + "01&ref_=adv_nxt"
        page = ImdbMovieScrapper(next_url)
        dataframe = page.MovieDataFrame()
    else:
        pass
data_frame_list.append(dataframe)

# --------------------CSV output-----------------------------
# Movie_Final_Dataframe = pd.concat(data_frame_list).to_csv('Movie_Final_Dataframe.csv')

# --------------------Console output-----------------------------
data_frame_list = pd.concat(data_frame_list)
print(tabulate(data_frame_list, headers='keys', tablefmt='fancy_grid'))
