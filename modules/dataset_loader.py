import os
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from itertools import product
from tqdm import tqdm
import math
 from sklearn.metrics.pairwise import cosine_similarity

from IPython.display import clear_output

colors = {
    'isolated': '#2E86AB',      # Blue
    'Optimized': '#A23B72',    # Purple
    'batch': '#F18F01'          # Orange
}

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

        # Initialize feature-related attributes
        self.has_features = False
        self.feature_map = {}
        self.feature_reverse_map = {}
        self.movie_to_features = {}     # movie_idx -> list of feature_idx

        if MOVIES_DIR is not None:
            self._load_movie_features(MOVIES_DIR)

    def _load_movie_features(self, MOVIES_DIR: str):
        print("Loading movie features...")

        with open(MOVIES_DIR, encoding="utf-8") as file:
            next(file)

            for line in tqdm(file, desc="Processing movie features"):
                line = line.strip()
                parts = line.split(',')
                if len(parts) < 3:
                    continue

                movie_id = int(parts[0])
                genres_str = parts[-1]

                if movie_id not in self.movies_map:
                    continue

                movie_idx = self.movies_map[movie_id]

                if genres_str and genres_str != "(no genres listed)":
                    genres = genres_str.split('|')

                    feature_indices = []
                    for genre in genres:
                        genre = genre.strip()
                        if genre not in self.feature_map:
                            feature_idx = len(self.feature_map)
                            self.feature_map[genre] = feature_idx
                            self.feature_reverse_map[feature_idx] = genre
                        else:
                            feature_idx = self.feature_map[genre]
                        
                        feature_indices.append(feature_idx)
                    
                    self.movie_to_features[movie_idx] = feature_indices

        self.__n_features = len(self.feature_map)
        self.has_features = self.__n_features > 0

        print(f"Loaded {self.__n_features} unique features (genres)")
        print(f"Features mapped for {len(self.movie_to_features)}/{self.__n_movies} movies")

    def _build_feature_matrix(self):
        movie_features = [[] for _ in range(self.__n_movies)]
        for movie_idx, feature_indices in self.movie_to_features.items():
            movie_features[movie_idx] = feature_indices
        return movie_features

    def compute_loss(self, mode="train", use_features=False):
        M, N = self.user_movie_counts()

        rating_errors = []
        total_ratings = 0.0

        if mode == 'train':
            users_idx = self.users[self.train_idx]
            movies_idx = self.movies[self.train_idx]
            ratings = self.ratings[self.train_idx]
        elif mode == "val":
            users_idx = self.users[self.val_idx]
            movies_idx = self.movies[self.val_idx]
            ratings = self.ratings[self.val_idx]
        elif mode == "test":
            users_idx = self.users[self.test_idx]
            movies_idx = self.movies[self.test_idx]
            ratings = self.ratings[self.test_idx]
        else:
            raise ValueError("Only three mode arguments allowed: train, val, test")

        pred_m_rated = self.mu + np.sum(self.V[movies_idx] * self.U[users_idx], axis=1) + self.BM[users_idx] + self.BN[movies_idx]
        rating_errors = (ratings - pred_m_rated) ** 2
        total_ratings = len(rating_errors)

        if total_ratings == 0:
            return 0.0, 0.0

        rating_error = np.sum(rating_errors)

        # RMSE is always just the data fit
        rmse = np.sqrt(rating_error / total_ratings) if total_ratings > 0 else 0.0

        # Loss differs by mode
        if mode == "train":
            users_norm = np.sum(self.U ** 2)
            movies_norm = np.sum(self.V ** 2)
            users_bias_squared = np.sum(self.BM ** 2)
            movies_bias_squared = np.sum(self.BN ** 2)

            loss = (self.lambda_ / 2) * rating_error + \
                    (self.gamma / 2) * (users_bias_squared + movies_bias_squared) + \
                    (self.tau / 2) * (users_norm)

            if use_features and self.has_features and hasattr(self, 'W'):
                feature_prior = self._compute_feature_prior()
                movie_deviation = np.sum((self.V - feature_prior)**2)
                feature_norm = np.sum(self.W**2)

                loss += (self.tau / 2) * movie_deviation + (self.eta / 2) * feature_norm
            else:
                loss += (self.tau / 2) * movies_norm
        else:
            # Test loss: only data fit (no regularization)
            loss = (self.lambda_ / 2) * rating_error  # or even just rating_error

        return loss, rmse

    def _compute_feature_prior(self):
        V_prior = np.zeros_like(self.V)

        for movie_idx in range(self.__n_movies):
            if movie_idx in self.movie_to_features:
                feature_indices = self.movie_to_features[movie_idx]
                if len(feature_indices) > 0:
                    V_prior[movie_idx] = np.mean(self.W[feature_indices], axis=0)
        
        return V_prior

    def train(self, test_size=0.1, latent_dim=10, n_iter=50, eval_inter=5, lambda_=1, tau=1, gamma=1, eta=1, use_features=False, biases_alone=False, verbose=True):
        self.lambda_ = lambda_
        self.tau = tau
        self.gamma = gamma
        self.eta = eta

        M, N  =self.user_movie_counts()

        if hasattr(self, 'train_idx'):
            pass
        else:
            self.train_idx,self.val_idx, self.test_idx = self.train_test_split(split_ratio = (1 - test_size)) 
            # Only work with TRAIN indices
            train_users = self.users[self.train_idx]
            train_movies = self.movies[self.train_idx]
            train_ratings = self.ratings[self.train_idx]

            # Pre-build lookups ONLY for train data
            print("Building lookup tables ...")
            self.user_to_indices = [[] for _ in range(M)] 
            self.movie_to_indices = [[] for _ in range(N)]

            for idx in tqdm(range(len(train_users)), desc="Building lookups"):
                self.user_to_indices[train_users[idx]].append(idx)
                self.movie_to_indices[train_movies[idx]].append(idx)

            # Convert to numpy arrays
            self.user_to_indices = [np.array(v, dtype=np.int32) if len(v) > 0 else np.array([], dtype=np.int32)
                            for v in self.user_to_indices]
            self.movie_to_indices = [np.array(v, dtype=np.int32) if len(v) > 0  else np.array([], dtype=np.int32)
                                for v in self.movie_to_indices]
            print("Lookup tables built")
            gc.collect()

        self.K = K = latent_dim
        self.U = np.random.randn(M, K) * 0.01
        self.V = np.random.randn(N, K) * 0.01
        self.BM = np.random.randn(M) * 0.01
        self.BN = np.random.randn(N) * 0.01
        self.mu = np.mean(self.ratings[self.train_idx])

        if use_features and self.has_features:
            self.W = np.random.randn(self.__n_features, K) * 0.01
            print(f"Using features: {self.__n_features} features with dim {K}")
        else:
            use_features = False
            print("Training without features")
        
        train_loss_history = []
        train_rmse_history = []
        val_loss_history = []
        val_rmse_history = []
        train_loss, train_rmse = self.compute_loss(mode="train")
        val_loss, val_rmse = self.compute_loss(mode="val")
        start_train_loss = train_loss
        start_val_loss = val_loss

        # Only work with TRAIN indices
        train_users = self.users[self.train_idx]
        train_movies = self.movies[self.train_idx]
        train_ratings = self.ratings[self.train_idx]

        if use_features:
            movie_features = self._build_feature_matrix()
            feature_to_movies = [[] for _ in range(self.__n_features)]
            for movie_idx, feat_indices in enumerate(movie_features):
                for feat_idx in feat_indices:
                    feature_to_movies[feat_idx].append(movie_idx)
            feature_to_movies = [np.array(v, dtype=np.int32) if len(v) > 0 else np.array([], dtype=np.int32) 
                                for v in feature_to_movies]
        if biases_alone is False:
            print("Updating Bias + User/Movie Embeeding")
        else:
            print("Updating Bias Alone")

        prev_val_loss = np.inf
        for epoch in tqdm(range(n_iter), total=n_iter, unit_scale=True, unit='it'):

            if biases_alone is False:
                # Update users latent factor - NO detrended_ratings array!
                for m in range(M):
                    indices = self.user_to_indices[m]
                    if len(indices) == 0:
                        continue

                    movies_rated_by_m = train_movies[indices]
                    ratings_by_m = train_ratings[indices]

                    # Detrend on-the-fly (no large array creation)
                    detrended = ratings_by_m - self.mu - self.BM[m] - self.BN[movies_rated_by_m]

                    V_rated = self.V[movies_rated_by_m]
                    numerator = lambda_ * np.sum(V_rated * detrended[:, np.newaxis], axis=0)
                    denominator = lambda_ * (V_rated.T @ V_rated) + (tau * np.eye(K))
                    self.U[m] = np.linalg.solve(denominator, numerator)

                # Update movies latent factor
                for n in range(N):
                    indices = self.movie_to_indices[n]
                    if len(indices) == 0:
                        continue

                    users_rating_movie = train_users[indices]
                    ratings_for_n = train_ratings[indices]

                    # Detrend on-the-fly
                    detrended = ratings_for_n - self.mu - self.BM[users_rating_movie] - self.BN[n]

                    U_rated = self.U[users_rating_movie]
                    numerator = lambda_ * np.sum(U_rated * detrended[:, np.newaxis], axis=0)

                    # Add feature prioir contribution if using prior
                    if use_features and n in self.movie_to_features:
                        feature_indices = self.movie_to_features[n]
                        if len(feature_indices) > 0:
                            v_prior = np.mean(self.W[feature_indices], axis=0)
                            numerator += tau * v_prior
                            denominator = lambda_ * (U_rated.T @U_rated) + (2 * tau *np.eye(K))
                        else:
                            denominator = lambda_ * (U_rated @ U_rated) + (tau * np.eye(K))
                    else:
                        denominator = lambda_ * (U_rated.T @ U_rated) + (tau * np.eye(K))
                    
                    self.V[n] = np.linalg.solve(denominator, numerator)

                # Update feature embeddings if using features
                if use_features:
                    for f in range(self.__n_features):
                        movies_with_feature = feature_to_movies[f]
                        if len(movies_with_feature) == 0:
                            continue

                        V_subset = self.V[movies_with_feature]
                        numerator = tau * np.sum(V_subset, axis=0)
                        denominator = tau * len(movies_with_feature) + eta
                        self.W[f] = numerator / denominator

            # Bias updates - compute predictions on-the-fly
            # User biases
            predictions = self.mu + np.sum(self.U[train_users] * self.V[train_movies], axis=1) + self.BN[train_movies]
            residuals = train_ratings - predictions
            per_user_residual_sum = np.bincount(train_users, weights=lambda_*residuals, minlength=M)
            per_user_count = np.bincount(train_users, minlength=M)
            mask = per_user_count > 0
            self.BM[mask] = per_user_residual_sum[mask] / (lambda_ * per_user_count[mask] + gamma)

            # Movie biases (recompute predictions with updated BM)
            predictions = self.mu + np.sum(self.U[train_users] * self.V[train_movies], axis=1) + self.BM[train_users]
            residuals = train_ratings - predictions
            per_movie_residual_sum = np.bincount(train_movies, weights=lambda_*residuals, minlength=N)
            per_movie_count = np.bincount(train_movies, minlength=N)
            mask = per_movie_count > 0
            self.BN[mask] = per_movie_residual_sum[mask] / (lambda_ * per_movie_count[mask] + gamma)

            # Evaluate
            train_loss, train_rmse = self.compute_loss(mode="train", use_features=use_features)
            train_loss_history.append(train_loss)
            train_rmse_history.append(train_rmse)

            val_loss, val_rmse = self.compute_loss(mode="val", use_features=use_features)
            val_loss_history.append(val_loss)
            val_rmse_history.append(val_rmse)

            if (epoch % eval_inter == 0) and (verbose):
                feat_str = " [with features]" if  use_features else ""
                print(f"Epoch {epoch}{feat_str}: Train Loss = {train_loss:.4f}, RMSE: {train_rmse:.4f} | Test Loss = {val_loss:.4f}, RMSE: {val_rmse:.4f}")

            if val_loss < prev_val_loss:
                prev_val_loss = val_loss
            else:
                break

        history = {
            "train_loss": train_loss_history, "val_loss": val_loss_history,
            "train_rmse": train_rmse_history, "val_rmse": val_rmse_history
        }
        
        if verbose:
            print(f"\nEND: Train Loss = {train_loss:.4f}, RMSE: {train_rmse:.6f} | Val Loss = {val_loss:.4f}, RMSE: {val_rmse:.6f}")
            print(f"Loss Reduction: Train {start_train_loss - train_loss:.4f}, Val {start_val_loss - val_loss:.4f}")

        return history

    def get_feature_embeddings(self):
        if hasattr(self, 'W'):
            return self.W, self.feature_map, self.feature_reverse_map
        return None, None, None

    def load_model(self, model_dir):
        hyper_params = np.load(model_dir)

        self.U = hyper_params['U']
        self.V = hyper_params['V']
        self.K = self.U.shape[-1]
        if 'W' in hyper_params:
            self.W = hyper_params['W']
        
        self.BM = hyper_params['BM']
        self.BN = hyper_params['BN']

        self.mu = hyper_params['mu']
        self.lambda_ = hyper_params['lambda_']
        self.gamma = hyper_params['gamma']
        self.tau = hyper_params['tau']
        if 'eta' in hyper_params:
            self.eta = hyper_params['eta']

    def get_hyperparameters(self):
        if hasattr(self, 'eta'):
            return {
                'mu': self.mu,
                'lambda_': self.lambda_,
                'gamma': self.gamma,
                'tau': self.tau,
                'eta': self.eta
            }
        else:
            return {
                'mu': self.mu,
                'lambda_': self.lambda_,
                'gamma': self.gamma,
                'tau': self.tau
            }

    def train_test_split(self, split_ratio: float):
        assert (split_ratio >= 0.0) and (split_ratio <= 1.0)
     
        # Initialize all four lists based on the size of users/movies
        train_idxs = []
        val_idxs = []
        test_idxs = []

        # Iterate over users
        for idx in tqdm(range(self.__n_entries), total=self.__n_entries):
            if random.uniform(0.0, 1.0) < split_ratio:
                train_idxs.append(idx)
            elif random.uniform(0.0, 1.0) <= (split_ratio + ((1 - split_ratio) / 2)):
                val_idxs.append(idx)
            else:
                test_idxs.append(idx)

        train_idxs = np.array(train_idxs)
        val_idxs = np.array(val_idxs)
        test_idxs = np.array(test_idxs)
        return train_idxs, val_idxs, test_idxs

    def user_movie_counts(self):
        return self.__n_users, self.__n_movies

    def train_val_performance(self, train_loss, train_rmse, val_loss, val_rmse, save_dir=None, save_name=None, title=None):
        fig = plt.figure(figsize=(14, 6))
        n_epochs = len(train_loss)
        assert n_epochs == len(train_rmse) == len(val_loss) == len(val_rmse)
        epochs = np.arange(1, n_epochs+1)

        # Train NLL
        plt.subplot(2, 2, 1)
        plt.plot(epochs, train_loss, alpha=0.3, linewidth=1, color=colors['Optimized'])
        plt.scatter(epochs, train_loss, label='Train NLL', s=30, color=colors['Optimized'], rasterized=True)
        plt.xscale('log')
        plt.ylabel("NLL", fontsize=11)
        plt.xlabel("Epoch", fontsize=11)
        plt.grid(alpha=0.3)
        plt.title("Train NLL", fontsize=12, fontweight='bold')

        # Test NLL
        plt.subplot(2, 2, 2)
        plt.plot(epochs, val_loss, alpha=0.3, linewidth=1, color=colors['Optimized'])
        plt.scatter(epochs, val_loss, label='Validation NLL', s=30, color=colors['Optimized'], rasterized=True)
        plt.xscale('log')
        plt.xlabel("Epoch", fontsize=11)
        plt.title("Validation NLL", fontsize=12, fontweight='bold')
        plt.grid(alpha=0.3)

        # Train RMSE
        plt.subplot(2, 2, 3)
        plt.plot(epochs, train_rmse, alpha=0.3, linewidth=1, color=colors['Optimized'])
        plt.scatter(epochs, train_rmse, label='Train RMSE', s=30, color=colors['Optimized'], rasterized=True)
        plt.xscale('log')
        plt.ylabel("RMSE", fontsize=11)
        plt.xlabel("Epoch", fontsize=11)
        plt.grid(alpha=0.3)
        plt.title("Train RMSE", fontsize=12, fontweight='bold')

        # Test RMSE
        plt.subplot(2, 2, 4)
        plt.plot(epochs, val_rmse, alpha=0.3, linewidth=1, color=colors['Optimized'])
        plt.scatter(epochs, val_rmse, label='Validation RMSE', s=30, color=colors['Optimized'], rasterized=True)
        plt.xscale('log')
        plt.xlabel("Epoch", fontsize=11)
        plt.title("Validation RMSE", fontsize=12, fontweight='bold')
        plt.grid(alpha=0.3)

        # Get legend from first subplot instead
        # ax1 = fig.get_axes()[0]
        # handles, labels_list = ax1.get_legend_handles_labels()

        if title is None:
            title  = "Bias + Latent-Factor Update Methods Comparison (32M Samples)"
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        # fig.legend(handles, labels_list, loc='upper center', bbox_to_anchor=(0.5, 0.93),
        #         ncol=3, frameon=True, fontsize=11, edgecolor='gray')

        plt.tight_layout(rect=[0, 0, 1, 0.90])  # Leave space for title and legendafdpi

        if save_dir is not None:
            if save_name is None:
                save_name = "train+val_performance"

            os.makedirs(f"{save_dir}/pdfs", exist_ok=True)
            os.makedirs(f"{save_dir}/pngs", exist_ok=True)
            fig.savefig(f"{save_dir}/pdfs/{save_name}.pdf", format="pdf", dpi=100, bbox_inches='tight')
            fig.savefig(f"{save_dir}/pngs/{save_name}.png", format="png", dpi=100, bbox_inches='tight')
    
    def test_performance(self, save_folder=None, save_name=None, title=None):
        test_NLL, test_RMSE = self.compute_loss(mode="test")
        test_performance = {
            'NLL (Log Scale)': np.log(test_NLL),
            'RMSE': test_RMSE
        }

        # Create figure and axis with a nice size
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

        # Data for plotting
        metrics = list(test_performance.keys())
        values = list(test_performance.values())

        # Define beautiful colors
        colors = ['#8b5cf6', '#06b6d4']  # Purple for NLL, Cyan for RMSE

        # Create bars with gradient effect
        bars = ax.bar(metrics, values, color=colors, width=0.6,
                        edgecolor='white', linewidth=2, alpha=0.9)

        # Add value labels on top of bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}',
                    ha='center', va='bottom', fontsize=14, fontweight='bold')

        # Styling
        ax.set_ylabel('Value', fontsize=14, fontweight='bold')
        if title is None:
            title = 'Test Performance Metrics'
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)

        # Add subtle background
        ax.set_facecolor('#f8f9fa')

        plt.tight_layout()

        if save_folder is not None:
            if save_name is None:
                save_name = "test_performance"

            os.makedirs(f"{save_folder}/pdfs", exist_ok=True)
            os.makedirs(f"{save_folder}/pngs", exist_ok=True)
            fig.savefig(f'{save_folder}/pdfs/{save_name}.pdf', format='pdf', dpi=100, bbox_inches='tight')
            fig.savefig(f'{save_folder}/pngs/{save_name}.pngs', format='png', dpi=100, bbox_inches='tight')

        plt.show()

    def plot_feature_embeddings(self, save_dir=None, save_name='genre_embeddings_2D', verbose=False):
        """
        Plot 2D embeddings of genre/feature vectors with labels.
        
        Parameters:
        -----------
        dataset : MovieLensDataset_Optimized_WithFeatures
            Trained dataset object with feature embeddings
        save_dir : str, optional
            Directory to save the plot
        save_name : str
            Name for the saved plot file
        """
        # Get feature embeddings
        W, feature_map, feature_reverse_map = self.get_feature_embeddings()
        
        if W is None:
            print("No feature embeddings found. Make sure you trained with use_features=True")
            return
        
        if W.shape[1] != 2:
            if verbose:
                print(f"Warning: Feature embeddings have {W.shape[1]} dimensions, not 2.")
                print("This visualization works best with 2D embeddings (latent_dim=2)")
            if W.shape[1] > 2:
                if verbose:
                    print("Using first 2 dimensions only...")
                W = W[:, :2]
            else:
                return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Extract coordinates
        x_coords = W[:, 0]
        y_coords = W[:, 1]
        
        # Plot points
        scatter = ax.scatter(x_coords, y_coords, 
                            s=150, 
                            c=range(len(W)), 
                            cmap='tab20',
                            alpha=0.7,
                            edgecolors='black',
                            linewidth=1.5,
                            zorder=3,
                            rasterized=True)
        
        # Add labels for each genre
        for feature_idx in range(len(W)):
            genre_name = feature_reverse_map[feature_idx]
            
            # Add text label with white background for readability
            ax.annotate(genre_name,
                    xy=(x_coords[feature_idx], y_coords[feature_idx]),
                    xytext=(5, 5),  # Offset text slightly
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='white', 
                            edgecolor='gray',
                            alpha=0.8),
                    zorder=4)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add axis lines through origin
        ax.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
        ax.axvline(x=0, color='k', linewidth=0.8, alpha=0.3)
        
        # Labels and title
        ax.set_xlabel('Latent Dimension 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latent Dimension 2', fontsize=12, fontweight='bold')
        ax.set_title('2D Embeddings of Genre Features', 
                    fontsize=14, 
                    fontweight='bold',
                    pad=20)
        
        # Add info text
        info_text = f"Number of genres: {len(W)}"
        ax.text(0.02, 0.98, info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save if directory provided
        if save_dir is not None:
            import os
            os.makedirs(f"{save_dir}/pdfs", exist_ok=True)
            os.makedirs(f"{save_dir}/pngs", exist_ok=True)
            
            fig.savefig(f"{save_dir}/pdfs/{save_name}.pdf", 
                    format="pdf", dpi=100, bbox_inches='tight')
            fig.savefig(f"{save_dir}/pngs/{save_name}.png", 
                    format="png", dpi=100, bbox_inches='tight')
            if verbose:
                print(f"Saved plots to {save_dir}")
        
        plt.show()
        
        return fig, ax


    def plot_feature_embeddings_with_clustering(self, save_dir=None, save_name='genre_embeddings_clustered', verbose=False):
        """
        Plot feature embeddings with visual clustering/grouping analysis.
        Shows which genres are embedded close together.
        
        Parameters:
        -----------
        dataset : MovieLensDataset_Optimized_WithFeatures
            Trained dataset object with feature embeddings
        save_dir : str, optional
            Directory to save the plot
        save_name : str
            Name for the saved plot file
        """
        W, feature_map, feature_reverse_map = self.get_feature_embeddings()
        
        if W is None or W.shape[1] != 2:
            print("Feature embeddings not available or not 2D")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 7))
        
        # Left plot: Basic embeddings
        ax1 = plt.subplot(1, 2, 1)
        x_coords = W[:, 0]
        y_coords = W[:, 1]
        
        scatter1 = ax1.scatter(x_coords, y_coords,
                            s=150,
                            c=range(len(W)),
                            cmap='tab20',
                            alpha=0.7,
                            edgecolors='black',
                            linewidth=1.5,
                            zorder=3,
                            rasterized=True)
        
        for feature_idx in range(len(W)):
            genre_name = feature_reverse_map[feature_idx]
            ax1.annotate(genre_name,
                        xy=(x_coords[feature_idx], y_coords[feature_idx]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='white',
                                edgecolor='gray',
                                alpha=0.8))
        
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
        ax1.axvline(x=0, color='k', linewidth=0.8, alpha=0.3)
        ax1.set_xlabel('Latent Dimension 1', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Latent Dimension 2', fontsize=11, fontweight='bold')
        ax1.set_title('Genre Embeddings', fontsize=12, fontweight='bold')
        
        # Right plot: Distance matrix / similarity
        ax2 = plt.subplot(1, 2, 2)
        
        # Compute pairwise distances
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(W, metric='euclidean'))
        
        # Plot heatmap
        im = ax2.imshow(distances, cmap='YlOrRd', aspect='auto')
        
        # Add genre labels
        genre_names = [feature_reverse_map[i] for i in range(len(W))]
        ax2.set_xticks(range(len(W)))
        ax2.set_yticks(range(len(W)))
        ax2.set_xticklabels(genre_names, rotation=90, fontsize=9)
        ax2.set_yticklabels(genre_names, fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Euclidean Distance', fontsize=10, fontweight='bold')
        
        ax2.set_title('Pairwise Genre Distances', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save if directory provided
        if save_dir is not None:
            import os
            os.makedirs(f"{save_dir}/pdfs", exist_ok=True)
            os.makedirs(f"{save_dir}/pngs", exist_ok=True)
            
            fig.savefig(f"{save_dir}/pdfs/{save_name}.pdf",
                    format="pdf", dpi=100, bbox_inches='tight')
            fig.savefig(f"{save_dir}/pngs/{save_name}.png",
                    format="png", dpi=100, bbox_inches='tight')
            if verbose:
                print(f"Saved clustered plots to {save_dir}")
        
        plt.show()
        
        if verbose:
            # Print some insights
            print("\n=== Genre Similarity Insights ===")
            print("Closest genre pairs:")
            for i in range(len(W)):
                for j in range(i+1, len(W)):
                    if distances[i, j] < np.percentile(distances, 10):  # Top 10% closest
                        print(f"  {genre_names[i]:20s} <-> {genre_names[j]:20s} : {distances[i,j]:.3f}")
            
            print("\nMost distant genre pairs:")
            for i in range(len(W)):
                for j in range(i+1, len(W)):
                    if distances[i, j] > np.percentile(distances, 90):  # Top 10% most distant
                        print(f"  {genre_names[i]:20s} <-> {genre_names[j]:20s} : {distances[i,j]:.3f}")
        
        return fig


    def plot_cosine_similarity_heatmap(self, save_dir=None, save_name='genre_cosine_similarity', z_score=False, verbose=False):
        """
        Plot heatmap of pairwise cosine similarities between genre embeddings.
        
        Parameters:
        -----------
        dataset : MovieLensDataset_Optimized_WithFeatures
            Trained dataset object with feature embeddings
        save_dir : str, optional
            Directory to save the plot
        save_name : str
            Name for the saved plot file
        """
        W, feature_map, feature_reverse_map = self.get_feature_embeddings()
        
        if W is None:
            print("No feature embeddings found.")
            return
        
        # Compute cosine similarity matrix
        cos_sim = cosine_similarity(W)
        if z_score:
            cos_sim = (cos_sim - cos_sim.mean()) / cos_sim.std()
        
        fig = plt.figure(figsize=(12, 10))
        genre_names = [feature_reverse_map[i] for i in range(len(W))]
        cos_sim_df = pd.DataFrame(cos_sim, index=genre_names, columns=genre_names)
        sns.clustermap(
            cos_sim_df,
            cmap='RdBu_r',
            figsize=(12, 10)
        )
        fig.tight_layout()
        
        # Save if directory provided
        if save_dir is not None:
            import os
            os.makedirs(f"{save_dir}/pdfs", exist_ok=True)
            os.makedirs(f"{save_dir}/pngs", exist_ok=True)
            
            fig.savefig(f"{save_dir}/pdfs/{save_name}.pdf",
                    format="pdf", dpi=100, bbox_inches='tight')
            fig.savefig(f"{save_dir}/pngs/{save_name}.png",
                    format="png", dpi=100, bbox_inches='tight')
            if verbose:
                print(f"Saved cosine similarity plot to {save_dir}")
        
        plt.show()

        if verbose:
            # Print insights
            print("\n" + "="*60)
            print("COSINE SIMILARITY ANALYSIS")
            print("="*60)
            
            # Find most similar pairs (excluding self-similarity)
            print("\nMost similar genre pairs (highest cosine similarity):")
            np.fill_diagonal(cos_sim, -2)  # Exclude diagonal
            top_pairs = []
            for i in range(len(W)):
                for j in range(i+1, len(W)):
                    top_pairs.append((i, j, cos_sim[i, j]))
            
            top_pairs.sort(key=lambda x: x[2], reverse=True)
            for i, j, sim in top_pairs[:10]:
                print(f"  {genre_names[i]:20s} <-> {genre_names[j]:20s} : {sim:.4f}")
            
            # Find most dissimilar pairs
            print("\nMost dissimilar genre pairs (lowest cosine similarity):")
            for i, j, sim in top_pairs[-10:]:
                print(f"  {genre_names[i]:20s} <-> {genre_names[j]:20s} : {sim:.4f}")
            
            print("="*60 + "\n")
        
        return fig, cos_sim


    def compute_genre_cosine_similarities(self, top_k=10, verbose=False):
        """
        Compute and display top-k most similar and dissimilar genre pairs
        based on cosine similarity.
        
        Parameters:
        -----------
        dataset : MovieLensDataset_Optimized_WithFeatures
            Trained dataset object with feature embeddings
        top_k : int
            Number of top pairs to display
        
        Returns:
        --------
        cos_sim : numpy array
            Full cosine similarity matrix
        similar_pairs : list
            List of (genre1, genre2, similarity) tuples (most similar)
        dissimilar_pairs : list
            List of (genre1, genre2, similarity) tuples (most dissimilar)
        """
        W, feature_map, feature_reverse_map = self.get_feature_embeddings()
        
        if W is None:
            print("No feature embeddings found.")
            return None, None, None
        
        # Compute cosine similarity
        cos_sim = cosine_similarity(W)
        
        genre_names = [feature_reverse_map[i] for i in range(len(W))]
        
        # Get all pairs (upper triangle, excluding diagonal)
        pairs = []
        for i in range(len(W)):
            for j in range(i+1, len(W)):
                pairs.append((genre_names[i], genre_names[j], cos_sim[i, j]))
        
        # Sort by similarity
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        similar_pairs = pairs[:top_k]
        dissimilar_pairs = pairs[-top_k:][::-1]  # Reverse to show least similar first

        if verbose:
            print("\n" + "="*70)
            print(f"TOP {top_k} MOST SIMILAR GENRE PAIRS (Cosine Similarity)")
            print("="*70)
            for g1, g2, sim in similar_pairs:
                bar = "█" * int(sim * 50)  # Visual bar
                print(f"{g1:20s} <-> {g2:20s} : {sim:6.4f} {bar}")
            
            print("\n" + "="*70)
            print(f"TOP {top_k} MOST DISSIMILAR GENRE PAIRS (Cosine Similarity)")
            print("="*70)
            for g1, g2, sim in dissimilar_pairs:
                bar = "█" * int((1 + sim) * 25)  # Scaled for negative values
                print(f"{g1:20s} <-> {g2:20s} : {sim:6.4f} {bar}")
            print("="*70 + "\n")
        
        return cos_sim, similar_pairs, dissimilar_pairs

class GridSearch:
    def __init__(self, param_grid):
        """
        param_grid: dict with parameter names as keys and lists of values
        Example: {'gamma': [0.0, 0.25], 'lambda': [0.5, 1.0]}
        """
        self.param_grid = param_grid
        self.results = []
    
    def fit(self, dataset:MovieLensDataset_Optimized):
        # Get all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = [self.param_grid[name] for name in param_names]

        total = math.prod(len(v) for v in param_values)
        
        for values in tqdm(product(*param_values), total=total, desc='Grid Search'):
            params = dict(zip(param_names, values))
            
            # Train model with these parameters
            _ = dataset.train(**params)
            loss, rmse = dataset.compute_loss(mode="val")
            
            self.results.append({
                'params': params,
                'score': loss,
                'rmse': rmse
            })
            
            print(f"Params: {params}, RMSE Score: {rmse:.4f}, NLL Score: {loss:.4f}")
            clear_output()
        
        # Find best parameters
        self.best_result = max(self.results, key=lambda x: x['score'])
        return self
    
    def best_params(self):
        return self.best_result['params']
    
    def best_score(self):
        return self.best_result['score']

    def best_rmse(self):
        return self.best_result['rmse']

    def get_results(self):
        return self.results