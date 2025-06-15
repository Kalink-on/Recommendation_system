import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import warnings
import os
from datetime import datetime


warnings.filterwarnings('ignore')

class MovieLensDataPreprocessor:
    def __init__(self, data_path='./', sample_fraction=None, dev_mode=False):
        self.dev_mode = dev_mode
        self.data_path = data_path
        self.sample_fraction = sample_fraction
        self.movies = None
        self.ratings = None
        self.tags = None
        self.processed_data = None

    def load_data(self):
        print(f"{datetime.now()} - Загрузка данных...")

        required_files = ['movies.csv', 'ratings.csv']
        for file in required_files:
            if not os.path.exists(f'{self.data_path}{file}'):
                raise FileNotFoundError(f"Файл {file} не найден в {self.data_path}")

        if self.sample_fraction:
            self.movies = pd.read_csv(f'{self.data_path}movies.csv')
            users = pd.read_csv(f'{self.data_path}ratings.csv', usecols=['userId']).drop_duplicates()
            sampled_users = users.sample(frac=self.sample_fraction)

            self.ratings = pd.read_csv(
                f'{self.data_path}ratings.csv',
                chunksize=100000,
                iterator=True
            )
            self.ratings = pd.concat([chunk[chunk['userId'].isin(sampled_users['userId'])]
                                    for chunk in self.ratings])

            self.tags = pd.read_csv(
                f'{self.data_path}tags.csv',
                chunksize=100000,
                iterator=True
            )
            self.tags = pd.concat([chunk[chunk['userId'].isin(sampled_users['userId'])]
                                 for chunk in self.tags])
        else:
            self.movies = pd.read_csv(f'{self.data_path}movies.csv')
            self.ratings = pd.read_csv(f'{self.data_path}ratings.csv', chunksize=1000000)
            self.ratings = pd.concat(self.ratings)
            self.tags = pd.read_csv(f'{self.data_path}tags.csv', chunksize=1000000)
            self.tags = pd.concat(self.tags)

        print(f"{datetime.now()} - Данные загружены. Фильмов: {len(self.movies)}, Оценок: {len(self.ratings)}")

    def clean_data(self):
        print(f"{datetime.now()} - Очистка данных...")
        self.movies['year'] = self.movies['title'].str.extract(r'\((\d{4})\)').astype('float')
        self.movies['clean_title'] = self.movies['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()
        self.movies['genres_list'] = self.movies['genres'].str.split('|')
        self.ratings['rating'] = self.ratings['rating'].astype('float32')
        self.ratings['userId'] = self.ratings['userId'].astype('int32')
        self.ratings['movieId'] = self.ratings['movieId'].astype('int32')
        valid_movies = self.movies['movieId'].unique()
        self.ratings = self.ratings[self.ratings['movieId'].isin(valid_movies)]
        print(f"{datetime.now()} - Данные очищены")

    def analyze_data(self):
        if not self.dev_mode:
            return

        print("\n=== ДИАГНОСТИКА (режим разработчика) ===")
        print(f"Фильмов: {len(self.movies)}, Оценок: {len(self.ratings)}")
        print(f"Топ-5 фильмов по популярности:\n{self.ratings['movieId'].value_counts().head(5)}")

        plt.figure(figsize=(10, 5))
        self.ratings['rating'].hist(bins=10, edgecolor='black')
        plt.title('Распределение рейтингов фильмов')
        plt.xlabel('Рейтинг')
        plt.ylabel('Количество оценок')
        plt.show()

        genre_counts = pd.Series(np.concatenate(self.movies['genres_list'].values)).value_counts()
        plt.figure(figsize=(12, 6))
        genre_counts.plot(kind='bar')
        plt.title('Распределение фильмов по жанрам')
        plt.xlabel('Жанр')
        plt.ylabel('Количество фильмов')
        plt.xticks(rotation=45)
        plt.show()

    def prepare_recommendation_data(self):
        print(f"{datetime.now()} - Подготовка рекомендательных данных...")
        if self.ratings is None or self.movies is None:
            raise ValueError("Данные не загружены. Сначала вызовите load_data()")

        valid_movies = set(self.movies['movieId']) & set(self.ratings['movieId'])
        self.ratings = self.ratings[self.ratings['movieId'].isin(valid_movies)]

        movie_stats = (
            self.ratings.groupby('movieId')
            .agg(avg_rating=('rating', 'mean'),
                 rating_count=('rating', 'count'))
            .reset_index()
        )
        movie_data = self.movies.merge(movie_stats, on='movieId', how='left')

        print(f"{datetime.now()} - Создание разреженной матрицы...")
        unique_users = self.ratings['userId'].unique()
        unique_movies = self.ratings['movieId'].unique()
        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        movie_to_idx = {movie: idx for idx, movie in enumerate(unique_movies)}

        rows = self.ratings['userId'].map(user_to_idx)
        cols = self.ratings['movieId'].map(movie_to_idx)
        values = self.ratings['rating'].values

        sparse_matrix = csr_matrix((values, (rows, cols)),
                                 shape=(len(unique_users), len(unique_movies)))

        cf_data = self.ratings[['userId', 'movieId', 'rating']].copy()

        prepared_data = {
            'movie_data': movie_data,
            'sparse_user_item': sparse_matrix,
            'user_mapping': user_to_idx,
            'movie_mapping': movie_to_idx,
            'ratings': cf_data,
            'movies': self.movies
        }

        print(f"{datetime.now()} - Данные подготовлены")
        return prepared_data

    def save_processed_data(self, filename='processed_data.pkl'):
        import pickle
        print(f"{datetime.now()} - Сохранение данных...")
        with open(filename, 'wb') as f:
            pickle.dump({
                'movie_data': self.movies,
                'ratings': self.ratings,
                'tags': self.tags
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"{datetime.now()} - Данные сохранены в {filename}")

    def run_pipeline(self):
        self.load_data()
        self.clean_data()
        self.analyze_data()
        prepared_data = self.prepare_recommendation_data()
        self.save_processed_data()
        return prepared_data

    def get_top_by_avg_rating(self, top_n=10, min_ratings=50):
        movie_stats = (
            self.ratings.groupby('movieId')
            .agg(avg_rating=('rating', 'mean'), rating_count=('rating', 'count'))
            .reset_index()
        )
        movie_data = self.movies.merge(movie_stats, on='movieId', how='left')
        filtered = movie_data[movie_data['rating_count'] >= min_ratings]
        top_movies = filtered.sort_values('avg_rating', ascending=False).head(top_n)
        return top_movies[['movieId', 'title', 'rating_count', 'avg_rating', 'genres']]

    def get_top_by_popularity(self, top_n=10, min_ratings=1):
        movie_stats = (
            self.ratings.groupby('movieId')
            .agg(rating_count=('rating', 'count'),
                 avg_rating=('rating', 'mean'))
            .reset_index()
        )
        movie_data = self.movies.merge(movie_stats, on='movieId', how='left')
        filtered = movie_data[movie_data['rating_count'] >= min_ratings]
        top_movies = filtered.sort_values('rating_count', ascending=False).head(top_n)
        return top_movies[['movieId', 'title', 'rating_count', 'avg_rating', 'genres']]

    def get_top_by_weighted_rating(self, top_n=10, min_ratings=50, m_parameter=None):
        movie_stats = (
            self.ratings.groupby('movieId')
            .agg(avg_rating=('rating', 'mean'),
                 rating_count=('rating', 'count'))
            .reset_index()
        )
        movie_data = self.movies.merge(movie_stats, on='movieId', how='left')
        filtered = movie_data[movie_data['rating_count'] >= min_ratings]
        C = filtered['avg_rating'].mean()
        m = filtered['rating_count'].quantile(0.90) if m_parameter is None else m_parameter
        filtered['weighted_rating'] = ((filtered['avg_rating'] * filtered['rating_count']) + (C * m)) / (
                    filtered['rating_count'] + m)
        top_movies = filtered.sort_values('weighted_rating', ascending=False).head(top_n)
        return top_movies[['movieId', 'title', 'weighted_rating', 'rating_count', 'avg_rating', 'genres']]

    def get_top_by_genre(self, genre, top_n=10, min_ratings=50, metric="weighted", m_parameter=None):
        genre_mask = self.movies["genres"].str.contains(genre, case=False, na=False)
        genre_movies = self.movies[genre_mask]
        if genre_movies.empty:
            raise ValueError(f"Нет фильмов в жанре '{genre}'")

        original_ratings = self.ratings
        original_movies = self.movies
        self.ratings = self.ratings[self.ratings["movieId"].isin(genre_movies["movieId"])]
        self.movies = genre_movies

        if metric == "weighted":
            result = self.get_top_by_weighted_rating(top_n, min_ratings, m_parameter)
            result = result.rename(columns={"weighted_rating": "rating_metric"})
        elif metric == "avg":
            result = self.get_top_by_avg_rating(top_n, min_ratings)
            result = result.rename(columns={"avg_rating": "rating_metric"})
        elif metric == "popularity":
            result = self.get_top_by_popularity(top_n, min_ratings)
            result = result.rename(columns={"rating_count": "rating_metric"})
        else:
            self.ratings = original_ratings
            self.movies = original_movies
            raise ValueError('Метрика должна быть "weighted", "avg" или "popularity"')

        self.ratings = original_ratings
        self.movies = original_movies
        if 'year' not in result.columns:
            result['year'] = result['title'].str.extract(r'\((\d{4})\)')
        return result[["movieId", "title", "year", "rating_metric", "rating_count", "genres"]]

    def get_top_by_year(self, year, top_n=10, min_ratings=50, metric="weighted", m_parameter=None):
        year_movies = self.movies[self.movies["year"] == year]
        if year_movies.empty:
            raise ValueError(f"Нет фильмов {year} года в данных")

        original_ratings = self.ratings
        original_movies = self.movies
        self.ratings = self.ratings[self.ratings["movieId"].isin(year_movies["movieId"])]
        self.movies = year_movies

        if metric == "weighted":
            result = self.get_top_by_weighted_rating(top_n, min_ratings, m_parameter)
            result = result.rename(columns={"weighted_rating": "rating_metric"})
        elif metric == "avg":
            result = self.get_top_by_avg_rating(top_n, min_ratings)
            result = result.rename(columns={"avg_rating": "rating_metric"})
        elif metric == "popularity":
            result = self.get_top_by_popularity(top_n, min_ratings)
            result = result.rename(columns={"rating_count": "rating_metric"})
        else:
            self.ratings = original_ratings
            self.movies = original_movies
            raise ValueError('Метрика должна быть "weighted", "avg" или "popularity"')

        self.ratings = original_ratings
        self.movies = original_movies
        result["year"] = year
        return result[["movieId", "title", "year", "rating_metric", "rating_count", "genres"]]
