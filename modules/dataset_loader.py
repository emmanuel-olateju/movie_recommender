from tqdm import tqdm

class MovieLensDataset:
    def __init__(self, CSV_DIR: str) -> None:

        self.users_map = dict()
        self.movies_map = dict()

        self.user_ratings = []
        self.movie_ratings = []


        with open(CSV_DIR, "r", encoding="utf-8") as file:
            next(file)

            for line in tqdm.tqdm(file, total=32000204):
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


            self.rows_reverse_map = {value: key for key, value in self.users_map.items()}
            self.columns_reverse_map = {value: key for key, value in self.movies_map.items()}

        self.__n_users = len(self.users_map)
        self.__n_movies = len(self.movies_map)