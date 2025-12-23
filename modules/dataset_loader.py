from tqdm import tqdm
import random
import numpy as np

class MovieLensDataset:
    def __init__(self, CSV_DIR: str, MOVIES_DIR: str=None, ) -> None:

        self.users_map = dict()
        self.movies_map = dict()

        self.users = []
        self.movies = []
        self.ratings = []


        with open(CSV_DIR, "r", encoding="utf-8") as file:
            next(file)

            for line in tqdm(file, total=32000204):
                line = line.strip(" ")
                values = line.split(",")
                # print(values)
                user = int(values[0])
                movie = int(values[1])
                rating = float(values[2])

                if int(rating) == 0:
                    continue

                if user not in self.users_map:
                    self.users_map[user] = len(self.users_map)
                if movie not in self.movies_map:
                    self.movies_map[movie] = len(self.movies_map)

                self.users.append(self.users_map[user])
                self.movies.append(self.movies_map[movie])
                self.ratings.append(rating)


            self.users_reverse_map = {value: key for key, value in self.users_map.items()}
            self.movies_reverse_map = {value: key for key, value in self.movies_map.items()}
        
        self.users = np.array(self.users)
        self.movies = np.array(self.movies)
        self.ratings = np.array(self.ratings)

        self.__n_users = len(self.users_map)
        self.__n_movies = len(self.movies_map)
        self.__n_entries = len(self.users)
        self.shape = (self.__n_users, self.__n_movies)

        if MOVIES_DIR is not None:
            self.__make_movie_features(MOVIES_DIR)

    def __getitem__(self, index):
        if isinstance(index, int):
            pass

    def __make_movie_features(self, CSV_DIR: str=None):

        self.movie_features_idx = dict()
        self.movie_features_map = dict()
        self.movie_features_reverse_map = dict()

        with open(CSV_DIR, encoding="utf-8") as file:
            next(file)

            for line in tqdm(file, desc="Making Movie features"):
                line = line.strip(" ")
                values = line.split(",")

                movie = int(values[0])
                title = int(values[1])
                genre = int(values[2])

                if title not in self.movie_features_map:
                    title_idx = len(self.movie_features_map)
                    self.movie_features_reverse_map[title] = title_idx
                    self.movie_features_reverse_map[title_idx] = title

                movie_idx = self.movies_map[movie]
                if movie_idx not in self.movie_features_idx:
                    self.movie_features_idx[movie_idx] = []
                self.movie_features_idx[movie_idx].append(title_idx)

                if genre not in self.movie_features_map:
                    genre_idx = len(self.movie_features_map)
                    self.movie_features_map[title] = genre_idx
                    self.movie_features_reverse_map[genre_idx] = genre

                    movie_idx = self.movies_map[movie]
                    if movie_idx not in self.movie_features_map:
                        self.movie_features_idx[movie_idx] = []
                    self.movie_features_idx[movie_idx].append(genre_idx)

    def train_test_split(self, split_ratio: float):
        assert (split_ratio >= 0.0) and (split_ratio <= 1.0)
     
        # Initialize all four lists based on the size of users/movies
        train_idxs = []
        test_idxs = []

        # Iterate over users
        for idx in tqdm(range(self.__n_entries), total=self.__n_entries):
            if random.uniform(0.0, 1.0) < split_ratio:
                train_idxs.append(idx)
            else:
                test_idxs.append(idx)
        train_idxs = np.array(train_idxs)
        test_idxs = np.array(test_idxs)
        return train_idxs, test_idxs

    def user_movie_counts(self):
        return self.__n_users, self.__n_movies

class Split:

    def __init__(ds, map, reverse_map):
        pass

class MoviesLensSplit:

    def __init__(self, ds: MovieLensDataset, split_ratio:0.9) -> None:
        assert (split_ratio >= 0.0) and (split_ratio <= 1.0)

        self.users_map = ds.users_map
        self.users_reverse_map = ds.users_reverse_map
        self.user_train = []
        self.user_test = []

        for user_key, user_idx in ds.users_map.items():

            self.user_train.append([])
            self.user_test.append([])

            for j, rating_tuple in enumerate(ds.user_ratings[self.user_idx]):
                if random.uniform(0.0, 1.0) >= split_ratio:
                    self.user_test[user_idx].append(rating_tuple)
                else:
                    self.user_train[user_idx].append(rating_tuple)

        
        self.movies_map = ds.movies_map
        self.movies_reverse_map = ds.movies_reverse_map
        self.movie_train = []
        self.movie_test = []

        for movie_key , movie_idx in ds.movie_map.items():

            self.movie_train.append([])
            self.movie_test.append([])

            for j, rating_tuple in enumerate(ds.movie_ratings[movie_idx]):
                if random.uniform(0.0, 1.0) >= split_ratio:
                    self.movie_test[movie_idx].append(rating_tuple)
                else:
                    self.movie_train[movie_idx].append(rating_tuple)


    def __getitem__(self, index: int):
        pass