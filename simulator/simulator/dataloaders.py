#!/usr/bin/env python3
from pathlib import Path
import datetime as dt

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


class SyntheticDataset(Dataset):
    def __init__(self, num_users, num_items, num_samples):
        self.num_users = num_users
        self.num_items = num_items
        self.num_samples = num_samples
        self.data = self.generate_synthetic_data()

    def generate_synthetic_data(self):
        # Generating synthetic data (user_id, item_id, rating)
        data = []
        for _ in range(self.num_samples):
            user_id = torch.randint(0, self.num_users, (1,))
            item_id = torch.randint(0, self.num_items, (1,))
            rating = 0.0 if torch.rand((1,)) <= 0.75 else 1.0
            data.append((user_id, item_id, rating))
        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # print(self.data[idx])
        return self.data[idx]


class MovieLensDataset(Dataset):
    def __init__(self, data_path):
        path = Path(data_path)
        if not path.is_dir():
            return ValueError(f"{data_path} is invalid")
        if any(
            not (path / name).exists()
            for name in ["u.info", "u.data", "u.user", "u.item"]
        ):
            return ValueError(f"{data_path} is missing ML files")
        with (path / "u.info").open() as info:
            metadata = info.readlines()
        self.num_users, self.num_items, self.num_samples = map(
            lambda x: int(x.split(" ")[0]), metadata
        )
        # print(self.num_users, self.num_items, self.num_samples)
        columns = ["user_id", "item_id", "rating", "timestamp"]
        self.data = pd.read_csv(
            path / "u.data", sep="\t", names=columns, usecols=range(3)
        )
        # print(path / "u.user")
        self.user = pd.read_csv(
            path / "u.user",
            sep="|",
            names=["user_id", "age", "gender", "occupation", "zip_code"],
        )
        age_en, gender_en, job_en = OneHotEncoder(), OneHotEncoder(), OneHotEncoder()
        self.user["age"] = (self.user["age"] / 5).round().astype(int) * 5 # round to nearest 5
        self.oh_user_age = age_en.fit_transform(self.user[["age"]]).toarray()
        print(f"{self.oh_user_age.shape=}")
        self.oh_user_gender = gender_en.fit_transform(self.user[["gender"]]).toarray()
        self.oh_user_job = job_en.fit_transform(self.user[["occupation"]]).toarray()

        movie_names = [
            "item_id",
            "title",
            "date",
            "release_date",
            "url",
            "unknown",
            "action",
            "adventure",
            "animation",
            "chrildrens",
            "comedy",
            "crime",
            "documentary",
            "drama",
            "fantasy",
            "film-noir",
            "horror",
            "musical",
            "mystery",
            "romance",
            "sci-fi",
            "thriller",
            "war",
            "western",
        ]

        movies = pd.read_csv(
            path / "u.item",
            sep="|",
            encoding="ISO-8859-1",
            names=movie_names,
            usecols=range(24),
        )
        movies = movies.drop("release_date", axis=1)
        movies = movies.drop("url", axis=1)
        movies["date"] = pd.to_datetime(movies["date"])
        movies["date_float"] = movies["date"].apply(self._date2float)
        # movies["date_float"] = (movies["date_float"] - movies["date_float"].min()) / (movies["date_float"].max() - movies["date_float"].min())
        
        minmax = MinMaxScaler((-1, 1))
        movies["date_float"] = minmax.fit_transform(movies[["date_float"]])
        movies = movies.dropna()
        self.genres = [
                "unknown",
                "action",
                "adventure",
                "animation",
                "chrildrens",
                "comedy",
                "crime",
                "documentary",
                "drama",
                "fantasy",
                "film-noir",
                "horror",
                "musical",
                "mystery",
                "romance",
                "sci-fi",
                "thriller",
                "war",
                "western",
            ]
        for genre in self.genres:
            minmax = MinMaxScaler((-1,1))
            movies[genre] = minmax.fit_transform(movies[[genre]])
        X = pd.DataFrame(
            movies,
            columns= ["date_float"] + self.genres,
        )
        y = movies["item_id"]
        self.action_classifier = KNeighborsClassifier(1)
        self.action_classifier.fit(X.values, y.values)
        self.items = movies
        max_rating = self.data["rating"].max()
        min_rating = self.data["rating"].min()
        threshold = round((max_rating - min_rating) * 0.75)
        # print(self.data.dtypes)
        # print(threshold)
        self.data["rating"] = self.data["rating"].mask(
            self.data["rating"] <= threshold, 0.0
        )
        self.data["rating"] = self.data["rating"].mask(
            self.data["rating"] > threshold, 1.0
        )
        # print(self.data["rating"].min(), self.data["rating"].max(), self.data["rating"].sum())
        # self.data = torch.tensor(self.data.to_numpy())

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        user_id = torch.tensor(self.data.iloc[idx, 0], dtype=torch.long)
        item_id = torch.tensor(self.data.iloc[idx, 1], dtype=torch.long)
        rating = torch.tensor(self.data.iloc[idx, 2], dtype=torch.float32)
        return user_id, item_id, rating

    def get_user_mean(self, idx):
        mask = self.data["user_id"].values == idx.item() + 1
        n = np.count_nonzero(mask)
        if n == 0:
            return 0
        umean = self.data[mask]["rating"].sum() / n
        assert umean >= 0, f"{umean=} {self.data[mask]['rating'].sum()=} {n=}"
        assert umean <= 1, f"{umean=} {self.data[mask]['rating'].sum()=} {n=}"
        return self.data[mask]["rating"].sum() / n

    def get_movie_mean(self, idx):
        mask = self.data["item_id"].values == idx + 1
        n = np.count_nonzero(mask)
        if n == 0:
            return 0
        return self.data[mask]["rating"].sum() / n

    def get_movie_genre(self, movie_id):
        mask = self.items["item_id"].values == movie_id

        # display(self.items[mask])
        if any(mask):
            return pd.DataFrame(self.items[mask], columns=self.genres).values.reshape(-1)
        return np.zeros(len(self.genres))

    # def get_user_info(self, user_id):

    def _date2float(self, dt_obj):
        return dt_obj.year - 1 + (dt_obj.month - 1) / 12 + (dt_obj.day - 1) / 365.25
