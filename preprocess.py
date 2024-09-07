from turtledemo.penrose import start

import pandas as pd
from datetime import datetime
import json

from skimage.data import data_dir
from unidecode import unidecode
import re
import os
import random
import math


def preprocess_ratings(in_path, out_dir):

    data = pd.read_csv(in_path)

    user_ratings = {}
    th_up = 3.3
    th_down = 2.7

    i = 0
    for row in data.iterrows():
        user_id = str(int(row[1][0]))
        movie_id = str(int(row[1][1]))
        rating = row[1][2]
        if rating > th_up:
            rating = 2
        elif rating < th_down:
            rating = 0
        else:
            rating = 1
        timestamp = row[1][3]
        timestamp = str(datetime.fromtimestamp(timestamp))
        obj = {
            "movie_id": movie_id,
            "rating": rating,
            "timestamp": timestamp
        }
        if user_id not in user_ratings:
            user_ratings[user_id] = [obj]
        else:
            user_ratings[user_id].append(obj)
        if i % 20000 == 0:
            print(i)
        i += 1

    for user_id in user_ratings:
        user_rating = sorted(user_ratings[user_id], key=lambda x: x['timestamp'])
        with open(f'{out_dir}/{user_id}.json', 'w') as f:
            json.dump(user_rating, f, indent=6)


def preprocess_tags(tags_path, out_path):
    tags_pd = pd.read_csv(tags_path)
    movie_tags = {}
    count = 0
    for tag_pd in tags_pd.iterrows():
        movie_id = str(int(tag_pd[1][1]))
        tag = unidecode(str(tag_pd[1][2])).upper()

        if movie_id not in movie_tags:
            movie_tags[movie_id] = {
                "tags": [{"tag": tag, "count": 1}],
                "tag_indices": {
                    tag : 0
                }
            }
        else:
            if tag not in movie_tags[movie_id]["tag_indices"]:
                tag_index = len(movie_tags[movie_id]["tags"])
                movie_tags[movie_id]['tags'].append({"tag": tag, "count": 1})
                movie_tags[movie_id]['tag_indices'][tag] = tag_index
            else:
                tag_index = movie_tags[movie_id]['tag_indices'][tag]
                movie_tags[movie_id]['tags'][tag_index]['count'] += 1
        if count % 1000 == 0:
            print(count)
        count += 1

    unique_tags = set()
    neu_movie_tags = {}
    for movie_id in movie_tags:
        print(movie_id)
        tags = movie_tags[movie_id]['tags']
        tags = sorted(tags, key=lambda x: x['count'], reverse=True)
        neu_movie_tags[movie_id] = []
        for i in range(min(len(tags), 5)):
            neu_movie_tags[movie_id].append(tags[i])
            unique_tags.add(tags[i]['tag'])

    out_data = {
        "unique_tags": list(unique_tags),
        "movie_tags": neu_movie_tags
    }

    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=6)


def preprocess_movies(movies_path, tags_path, out_dir):
    movies_pd = pd.read_csv(movies_path)

    unique_years = set()
    unique_genres = set()
    movies = {}
    count = 0
    for movie_pd in movies_pd.iterrows():
        movie_id = str(int(movie_pd[1][0]))
        movie_name = movie_pd[1][1]
        matches = re.findall(r'\((\d{4})\)', movie_name)
        try:
            year = int(matches[-1])
        except:
            year = None
        unique_years.add(year)
        movie_genres = movie_pd[1][2].split("|")
        for genre in movie_genres:
            unique_genres.add(genre)

        movies[movie_id] = {
            "name": movie_name,
            "year": year,
            "genres": movie_genres,
        }
        if count % 1000 == 0:
            print(count)
        count += 1

    with open(tags_path, 'r') as f:
        tags = json.load(f)

    unique_tags = tags['unique_tags']
    unique_genres = list(unique_genres)
    unique_years = list(unique_years)

    for movie_id in movies:
        movie = movies[movie_id]
        try:
            movie_tags = tags['movie_tags'][movie_id]
        except:
            movie_tags = []
        neu_movie = {
            "movie_id": int(movie_id),
            "name": movie['name'],
            "year": movie['year'],
            "year_index": unique_years.index(movie['year']),
            "genres": movie['genres'],
            "genre_indices": [],
            "tags": [],
            "tag_indices": [],
            "tag_counts": []
        }
        for tag in movie_tags:
            neu_movie['tag_counts'].append(tag['count'])
            neu_movie['tag_indices'].append(unique_tags.index(tag['tag']))
            neu_movie['tags'].append(tag['tag'])
        for genre in movie['genres']:
            neu_movie['genre_indices'].append(unique_genres.index(genre))

        with open(f'{out_dir}/{movie_id}.json', 'w') as f:
            json.dump(neu_movie, f, indent=6)


def prepare_dataset(ratings_dir, out_dir, sample_divider=30, past_movie_count=10, future_movie_count=5, random_seed=42):
    # user_id|sampled_index<past>movie_id-rating-movie_id-rating<future>movie_id-movie_id-movie_id<future_mask>movie_id-movie_id<past_mask>movie_id-movie_id
    random.seed(random_seed)

    if sample_divider < past_movie_count + future_movie_count:
        sample_divider = past_movie_count + future_movie_count

    for n, name in enumerate(os.listdir(ratings_dir)):
        if n % 5000 == 0:
            print(n)
        if name == '.DS_Store':
            continue
        user_id = name.split('.')[0]
        with open(f'{ratings_dir}/{name}', 'r') as f:
            data = json.load(f)
        required_data_count = 0
        future_selectable_count = 0
        for i in range(1, len(data)+1):
            if data[-i]['rating'] == 2:
                future_selectable_count += 1
            required_data_count += 1
            if future_selectable_count >= future_movie_count:
                break
        required_data_count += past_movie_count
        if len(data) < required_data_count:
            continue
        for i in range(math.ceil(len(data)/sample_divider)):
            out_path = f"{out_dir}/{user_id}_{i}.txt"
            datum = f"{user_id}|{i}<past>"
            past_mask = "<past_mask>"
            if len(data) == required_data_count:
                start_index = 0
            else:
                start_index = random.randint(0, len(data)-required_data_count-1)
            for j in range(start_index):
                rating = data[j]
                past_mask += f'{rating["movie_id"]}-'
            for j in range(past_movie_count):
                rating = data[start_index+j]
                datum += f'{rating["movie_id"]}-{rating["rating"]}-'
                past_mask += f'{rating["movie_id"]}-'
            datum = datum[:-1] + "<future>"
            selected_movie_count = 0
            current_index = start_index + past_movie_count
            while selected_movie_count < future_movie_count and current_index < len(data):
                rating = data[current_index]
                if rating['rating'] == 2:
                    selected_movie_count += 1
                    datum += rating['movie_id'] + "-"
                current_index += 1
            datum = datum[:-1] + "<future_mask>"
            future_mask_added = False
            while current_index < len(data):
                rating = data[current_index]
                if rating['rating'] == 2:
                    datum += rating['movie_id'] + "-"
                    future_mask_added = True
                current_index += 1
            if future_mask_added:
                datum = datum[:-1] + past_mask[:-1]
            else:
                datum = datum + past_mask[:-1]
            with open(out_path, 'w') as f:
                f.write(datum)


def find_aux_data(movies_dir, out_path):
    # genre count +
    # movie count +
    # movie id mapping to index +
    # tag count +
    # year count +

    unique_movie_ids = set()
    unique_genres = set()
    unique_tags = set()
    unique_years = set()
    for name in os.listdir(movies_dir):
        if name == '.DS_Store':
            continue
        with open(f"{movies_dir}/{name}", 'r') as f:
            data = json.load(f)
        unique_movie_ids.add(data['movie_id'])
        for genre in data['genre_indices']:
            unique_genres.add(genre)
        for tag in data['tag_indices']:
            unique_tags.add(tag)
        unique_years.add(data['year_index'])

    out_data = {
        "genre_count": len(unique_genres),
        "tag_count": len(unique_tags),
        "year_count": len(unique_years),
        "movie_count": len(unique_movie_ids),
        "movie_indices": list(unique_movie_ids)
    }

    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=6)


if __name__ == '__main__':
    # preprocess_ratings('../movie_recommender_data/ml-32m/ratings.csv', '../movie_recommender_data/ratings')
    # preprocess_tags('../movie_recommender_data/ml-32m/tags.csv', '../movie_recommender_data/ml-32m/tags.json')
    # preprocess_movies("../movie_recommender_data/ml-32m/movies.csv", '../movie_recommender_data/ml-32m/tags.json', '../movie_recommender_data/movies')
    prepare_dataset('../movie_recommender_data/ratings', '../movie_recommender_data/dataset')
    # find_aux_data("../movie_recommender_data/movies", '../movie_recommender_data/aux_data.json')
