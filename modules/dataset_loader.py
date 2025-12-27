from tqdm import tqdm
import random
import numpy as np

class MovieLensDataset_Base:
    def __init__(self, CSV_DIR: str, MOVIES_DIR: str=None, ) -> None:

        self.users_map = dict()
        self.movies_map = dict()

        self.user_ratings = []
        self.movie_ratings = []


        with open(CSV_DIR, "r", encoding="utf-8") as file:
            next(file)

            for line in tqdm(file, total=32000204):
                line = line.strip(" ")
                values = line.split(",")
                # print(values)
                user = int(values[0])
                movie = int(values[1])
                rating = float(values[2])

                if user not in self.users_map:
                    self.users_map[user] = len(self.users_map)
                    self.user_ratings.append([])
                if movie not in self.movies_map:
                    self.movies_map[movie] = len(self.movies_map)
                    self.movie_ratings.append([])

                self.user_ratings[self.users_map[user]].append((rating, self.movies_map[movie]))
                self.movie_ratings[self.movies_map[movie]].append((rating, self.users_map[user]))


            self.users_reverse_map = {value: key for key, value in self.users_map.items()}
            self.movies_reverse_map = {value: key for key, value in self.movies_map.items()}

        self.__n_users = len(self.users_map)
        self.__n_movies = len(self.movies_map)

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
        user_train = [[] for _ in range(len(self.users_map))]
        user_test = [[] for _ in range(len(self.users_map))]
        movie_train = [[] for _ in range(len(self.movies_map))]
        movie_test = [[] for _ in range(len(self.movies_map))]

        # Iterate over users
        for user_key, user_idx in self.users_map.items():
            # Iterate over all ratings for this user
            for rating, movie_idx in self.user_ratings[user_idx]:

                # --- Perform the split DECISION ONCE ---
                if random.uniform(0.0, 1.0) < split_ratio:
                    # TRAIN SET

                    # 1. Update user-centric train list
                    user_train[user_idx].append((rating, movie_idx))

                    # 2. Update movie-centric train list
                    # Note: movie_ratings usually stores (rating, user_idx)
                    movie_train[movie_idx].append((rating, user_idx))
                else:
                    # TEST SET

                    # 1. Update user-centric test list
                    user_test[user_idx].append((rating, movie_idx))

                    # 2. Update movie-centric test list
                    movie_test[movie_idx].append((rating, user_idx))

        return user_train, user_test, movie_train, movie_test

class MovieLensDataset_Optimized:
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

    def compute_loss(self, mode="train"):
        pass

    def train(self, test_size=0.1, latent_dim=10, n_iter=50, lambda_=1, tau=1, gamma=1):
       self.train_idx, self.test_idx = self.train_test_split(split_ratio=1 - test_size) 
       M, N  =self.user_movie_counts()

       self.K = K = latent_dim
       self.U = np.random.randn(M, K)
       self.V = np.random.randn(N, K)
       self.BM = np.random.randn(M)
       self.BN = np.random.randn(N)
       self.mu = np.mean(self.ratings[self.train_idx])
       
       train_loss_history = []
       train_rmse_history = []
       test_loss_history = []
       test_rmse_history = []
       train_loss, train_rmse = self.compute_loss()
       test_loss, test_rmse = self.compute_loss(mode="test")

       # Only work with TRAIN indices
       train_users = self.users[self.train_idx]
       train_movies = self.movies[self.train_idx]
       train_ratings = self.ratings[self.train_idx]


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

    def __init__(self, ds: MovieLensDataset_Base, split_ratio:0.9) -> None:
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