# MovieRecommender-DoEL

Main goal of this project is to create a movie recommendation system
for me and my gf. Sometimes we hustle to find worthy movie to spend time with.
Because I want to solve my problem, I designed the requirements according to
my needs. 
1. We can not use 5-star rating system. We might say we liked/disliked or we are natural to a movie. This constraint might be narrowed if we decide not to do any ratings.
2. The model shouldn't suggest previously watched movies.
3. The main purpose of the model is not to make recommendations to the people included in the dataset meaning it should act properly whenever new user is introduced. So, users are dynamic meaning static user encoding system can not be built. Also, the model shouldn't take user data rather than the previously watched 10 movies.
4. It should suggest most likable movies. It shouldn't suggest whether we would like or dislike a movie. So, it should make predictions on top of all the movies, not for explicitly one movie.


# Version 0

## V0.1.0
### Current Plan
* Change dataset structure
  * 10 past movie information with ratings
  * 5 future positive movies
  * 5 future negative and neutral movies (negatives have priority)
* Change encoder
* Update model and loss mechanism


## v0.0.0

### 2024.09.07

* Initial Experiments are completed
  * Because the model tries to predict 5 positive classes out of 80k classes, it gets stuck predicting all of the classes as negative.
  * Training scheme will be changed

### 2024.09.06

* Created encoder
* Split dataset
* Implemented loader
* Created initial DL structure

### 2024.09.05
* Retrieved [MovieLens 32M](https://grouplens.org/datasets/movielens/) dataset.
* Handled preprocess operations arranging movie and rating information appropriate to next steps and requirements.

