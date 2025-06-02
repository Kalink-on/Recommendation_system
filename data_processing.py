import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')


class MovieLensDataPreprocessor:
    def __init__(self, data_path='./', sample_fraction=None):
        """
        Инициализация с возможностью семплирования данных

        Parameters:
        - data_path: путь к папке с данными
        - sample_fraction: доля данных для семплирования (None для всех данных)
        """
        self.data_path = data_path
        self.sample_fraction = sample_fraction
        self.movies = None
        self.ratings = None
        self.tags = None
        self.processed_data = None

    def load_data(self):
        """Загрузка данных с возможностью семплирования"""
        print(f"{datetime.now()} - Загрузка данных...")

        # Загрузка с семплированием если нужно
        if self.sample_fraction:
            self.movies = pd.read_csv(f'{self.data_path}movies.csv')

            # Семплируем пользователей чтобы сохранить связи в данных
            users = pd.read_csv(f'{self.data_path}ratings.csv', usecols=['userId']).drop_duplicates()
            sampled_users = users.sample(frac=self.sample_fraction)

            # Загружаем только рейтинги для выбранных пользователей
            self.ratings = pd.read_csv(
                f'{self.data_path}ratings.csv',
                chunksize=100000,
                iterator=True
            )
            self.ratings = pd.concat([chunk[chunk['userId'].isin(sampled_users['userId'])]
                                      for chunk in self.ratings])

            # Аналогично для тегов
            self.tags = pd.read_csv(
                f'{self.data_path}tags.csv',
                chunksize=100000,
                iterator=True
            )
            self.tags = pd.concat([chunk[chunk['userId'].isin(sampled_users['userId'])]
                                   for chunk in self.tags])
        else:
            # Полная загрузка данных
            self.movies = pd.read_csv(f'{self.data_path}movies.csv')

            # Загрузка по частям для экономии памяти
            self.ratings = pd.read_csv(f'{self.data_path}ratings.csv', chunksize=1000000)
            self.ratings = pd.concat(self.ratings)

            self.tags = pd.read_csv(f'{self.data_path}tags.csv', chunksize=1000000)
            self.tags = pd.concat(self.tags)

        print(f"{datetime.now()} - Данные загружены. Фильмов: {len(self.movies)}, Оценок: {len(self.ratings)}")

    def clean_data(self):
        """Оптимизированная очистка данных"""
        print(f"{datetime.now()} - Очистка данных...")

        # Обработка фильмов (оптимизированная)
        self.movies['year'] = self.movies['title'].str.extract(r'\((\d{4})\)').astype('float')
        self.movies['clean_title'] = self.movies['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()

        # Разделение жанров - сохраняем как список для экономии памяти
        self.movies['genres_list'] = self.movies['genres'].str.split('|')

        # Оптимизация типов данных
        self.ratings['rating'] = self.ratings['rating'].astype('float32')
        self.ratings['userId'] = self.ratings['userId'].astype('int32')
        self.ratings['movieId'] = self.ratings['movieId'].astype('int32')

        # Оставляем только фильмы, которые есть в movies.csv
        valid_movies = self.movies['movieId'].unique()
        self.ratings = self.ratings[self.ratings['movieId'].isin(valid_movies)]

        print(f"{datetime.now()} - Данные очищены")

    def analyze_data(self):
        """Оптимизированный анализ данных"""
        print(f"{datetime.now()} - Анализ данных...")

        # Распределение рейтингов (по частям для экономии памяти)
        plt.figure(figsize=(10, 5))
        self.ratings['rating'].hist(bins=10, edgecolor='black')
        plt.title('Распределение рейтингов фильмов')
        plt.xlabel('Рейтинг')
        plt.ylabel('Количество оценок')
        plt.show()

        # Топ фильмов (вычисляем через groupby с агрегацией)
        top_movies = (
            self.ratings.groupby('movieId')
            .agg(rating_count=('rating', 'count'), avg_rating=('rating', 'mean'))
            .sort_values('rating_count', ascending=False)
            .head(10)
            .merge(self.movies, on='movieId')
        )

        print("\nТоп-10 фильмов по количеству оценок:")
        print(top_movies[['movieId', 'title', 'rating_count', 'avg_rating']])

        # Распределение жанров (оптимизированный подсчет)
        genre_counts = pd.Series(
            np.concatenate(self.movies['genres_list'].values)
        ).value_counts()

        plt.figure(figsize=(12, 6))
        genre_counts.plot(kind='bar')
        plt.title('Распределение фильмов по жанрам')
        plt.xlabel('Жанр')
        plt.ylabel('Количество фильмов')
        plt.xticks(rotation=45)
        plt.show()

    def prepare_recommendation_data(self):
        """Подготовка данных с использованием разреженных матриц"""
        print(f"{datetime.now()} - Подготовка рекомендательных данных...")

        # 1. Подготовка данных для популярных рекомендаций
        movie_stats = (
            self.ratings.groupby('movieId')
            .agg(avg_rating=('rating', 'mean'),
                 rating_count=('rating', 'count'))
            .reset_index()
        )
        movie_data = self.movies.merge(movie_stats, on='movieId', how='left')

        # 2. Создание разреженной user-item матрицы
        print(f"{datetime.now()} - Создание разреженной матрицы...")

        # Получаем уникальные id пользователей и фильмов
        unique_users = self.ratings['userId'].unique()
        unique_movies = self.ratings['movieId'].unique()

        # Создаем маппинги для компактного представления
        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        movie_to_idx = {movie: idx for idx, movie in enumerate(unique_movies)}

        # Создаем разреженную матрицу
        rows = self.ratings['userId'].map(user_to_idx)
        cols = self.ratings['movieId'].map(movie_to_idx)
        values = self.ratings['rating'].values

        sparse_matrix = csr_matrix((values, (rows, cols)),
                                   shape=(len(unique_users), len(unique_movies)))

        # 3. Подготовка данных для коллаборативной фильтрации
        cf_data = self.ratings[['userId', 'movieId', 'rating']].copy()

        # Сохранение подготовленных данных
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
        """Сохранение данных с оптимизацией памяти"""
        import pickle
        print(f"{datetime.now()} - Сохранение данных...")

        # Для больших данных лучше использовать сжатие
        with open(filename, 'wb') as f:
            pickle.dump({
                'movie_data': self.movies,
                'ratings': self.ratings,
                'tags': self.tags
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"{datetime.now()} - Данные сохранены в {filename}")

    def run_pipeline(self):
        """Запуск всего пайплайна"""
        self.load_data()
        self.clean_data()
        self.analyze_data()
        prepared_data = self.prepare_recommendation_data()
        self.save_processed_data()
        return prepared_data

    def get_top_by_avg_rating(self, top_n=10, min_ratings=50):
        """
        Возвращает топ-N фильмов по среднему рейтингу

        - top_n: количество фильмов в топе
        - min_ratings: минимальное количество оценок для включения в топ

        На выходе:
        - DataFrame с колонками: movieId, title, rating_count, avg_rating, genres
        """

        # Получаем данные о фильмах с рейтингами
        movie_stats = (
            self.ratings.groupby('movieId')
            .agg(avg_rating=('rating', 'mean'), rating_count=('rating', 'count'))
            .reset_index()
        )

        # Объединяем с основной информацией о фильмах
        movie_data = self.movies.merge(movie_stats, on='movieId', how='left')

        # Фильтруем по минимальному количеству оценок
        filtered = movie_data[movie_data['rating_count'] >= min_ratings]

        # Сортируем по убыванию рейтинга и берем топ-N
        top_movies = filtered.sort_values('avg_rating', ascending=False).head(top_n)

        return top_movies[['movieId', 'title', 'rating_count', 'avg_rating', 'genres']]

    def get_top_by_popularity(self, top_n=10, min_ratings=1):
        """
        Возвращает топ-N фильмов по популярности (количеству оценок)

        На выходе:
        - DataFrame с колонками: movieId, title, rating_count, avg_rating, genres
        """
        # Получаем данные о фильмах с количеством оценок
        movie_stats = (
            self.ratings.groupby('movieId')
            .agg(rating_count=('rating', 'count'),
                 avg_rating=('rating', 'mean'))
            .reset_index()
        )

        # Объединяем с основной информацией о фильмах
        movie_data = self.movies.merge(movie_stats, on='movieId', how='left')

        # Фильтруем по минимальному количеству оценок
        filtered = movie_data[movie_data['rating_count'] >= min_ratings]

        # Сортируем по убыванию количества оценок и берем топ-N
        top_movies = filtered.sort_values('rating_count', ascending=False).head(top_n)

        return top_movies[['movieId', 'title', 'rating_count', 'avg_rating', 'genres']]

    def get_top_by_weighted_rating(self, top_n=10, min_ratings=50, m_parameter=None):
        """
        Возвращает топ-N фильмов по взвешенному рейтингу (Bayesian Average)

        - top_n: количество фильмов в топе
        - min_ratings: минимальное количество оценок для включения в топ
        - m_parameter: минимальное число оценок для доверия (если None, берется 90-й перцентиль)

        На выход:
        - DataFrame с колонками: movieId, title, weighted_rating, rating_count, avg_rating, genres
        """
        # 1. Считаем средний рейтинг и количество оценок для каждого фильма
        movie_stats = (
            self.ratings.groupby('movieId')
            .agg(avg_rating=('rating', 'mean'),
                 rating_count=('rating', 'count'))
            .reset_index()
        )

        # 2. Объединяем с названиями фильмов
        movie_data = self.movies.merge(movie_stats, on='movieId', how='left')

        # 3. Фильтруем фильмы с min_ratings
        filtered = movie_data[movie_data['rating_count'] >= min_ratings]

        # 4. Вычисляем глобальный средний рейтинг (C)
        C = filtered['avg_rating'].mean()

        # 5. Определяем параметр m (минимальное количество оценок для доверия)
        if m_parameter is None:
            m = filtered['rating_count'].quantile(0.90)  # 90-й перцентиль
        else:
            m = m_parameter

        # 6. Вычисляем взвешенный рейтинг (Bayesian Average)
        filtered['weighted_rating'] = ((filtered['avg_rating'] * filtered['rating_count']) + (C * m)) / (filtered['rating_count'] + m)

        # 7. Сортируем по взвешенному рейтингу и берем топ-N
        top_movies = filtered.sort_values('weighted_rating', ascending=False).head(top_n)

        return top_movies[['movieId', 'title', 'weighted_rating', 'rating_count', 'avg_rating', 'genres']]

    def get_top_by_genre(self, genre, top_n=10, min_ratings=50, metric="weighted", m_parameter=None ):
        """
        Возвращает топ-N фильмов указанного жанра по выбранной метрике, используя существующие методы.

        Parameters:
        - genre: жанр (например, "Comedy", "Action")
        - top_n: количество фильмов в топе
        - min_ratings: минимальное количество оценок
        - metric: критерий сортировки ("weighted", "avg", "popularity")
        - m_parameter: параметр m для взвешенного рейтинга (если None - 90-й перцентиль)

        Returns:
        - DataFrame с колонками: movieId, title, rating_metric, rating_count, genres
        """
        # 1. Фильтрация по жанру
        genre_mask = self.movies["genres"].str.contains(genre, case=False, na=False)
        genre_movies = self.movies[genre_mask]

        if genre_movies.empty:
            raise ValueError(f"Нет фильмов в жанре '{genre}'")

        # 2. Сохраняем исходные данные
        original_ratings = self.ratings
        original_movies = self.movies

        # 3. Подменяем данные
        self.ratings = self.ratings[self.ratings["movieId"].isin(genre_movies["movieId"])]
        self.movies = genre_movies

        # 4. Выбираем метод
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
            # Восстанавливаем данные ПЕРЕД возвратом ошибки
            self.ratings = original_ratings
            self.movies = original_movies
            raise ValueError('Метрика должна быть "weighted", "avg" или "popularity"')

        # 5. Восстанавливаем данные перед возвратом результата
        self.ratings = original_ratings
        self.movies = original_movies

        return result[["movieId", "title", "rating_metric", "rating_count", "genres"]]

    def get_top_by_year(self, year, top_n=10, min_ratings=50, metric="weighted", m_parameter=None):
        """
        Возвращает топ-N фильмов указанного года по выбранной метрике, используя существующие методы.

        Parameters:
        - year: год выпуска фильма (int)
        - top_n: количество фильмов в топе
        - min_ratings: минимальное количество оценок
        - metric: критерий сортировки ("weighted", "avg", "popularity")
        - m_parameter: параметр m для взвешенного рейтинга (если None - 90-й перцентиль)

        Returns:
        - DataFrame с колонками: movieId, title, year, rating_metric, rating_count, genres
        """
        # 1. Фильтрация по году
        year_movies = self.movies[self.movies["year"] == year]

        if year_movies.empty:
            raise ValueError(f"Нет фильмов {year} года в данных")

        # 2. Сохраняем исходные данные
        original_ratings = self.ratings
        original_movies = self.movies

        # 3. Подменяем данные
        self.ratings = self.ratings[self.ratings["movieId"].isin(year_movies["movieId"])]
        self.movies = year_movies

        # 4. Выбираем метод
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
            # Восстанавливаем данные ПЕРЕД возвратом ошибки
            self.ratings = original_ratings
            self.movies = original_movies
            raise ValueError('Метрика должна быть "weighted", "avg" или "popularity"')

        # 5. Восстанавливаем данные перед возвратом результата
        self.ratings = original_ratings
        self.movies = original_movies

        # Добавляем год в результат
        result["year"] = year

        return result[["movieId", "title", "year", "rating_metric", "rating_count", "genres"]]


if __name__ == "__main__":
    # Пример использования с семплированием 10% данных
    print("Запуск обработки данных...")
    preprocessor = MovieLensDataPreprocessor(
        data_path='ml-latest/',
        sample_fraction=0.1  # Используем 10% данных для демонстрации
    )

    prepared_data = preprocessor.run_pipeline()

    top_movies = preprocessor.get_top_by_avg_rating(top_n=10, min_ratings=50)
    print("\nТоп-10 фильмов по среднему рейтингу:")
    print(top_movies)

    top_by_popularity = preprocessor.get_top_by_popularity(top_n=10, min_ratings=50)
    print("\nТоп-10 фильмов по популярности (количеству оценок):")
    print(top_by_popularity)

    top_weighted = preprocessor.get_top_by_weighted_rating(top_n=10, min_ratings=50)
    print("\nТоп-10 фильмов по взвешенному рейтингу:")
    print(top_weighted)

    top_comedy = preprocessor.get_top_by_genre(genre="Comedy", top_n=5, min_ratings=50,metric="weighted")
    print("\nТоп-5 фильмов в жанре комедия по взвешенному рейтингу:")
    print(top_comedy)

    top_comedy = preprocessor.get_top_by_genre(genre="Comedy", top_n=5, min_ratings=50, metric="avg")
    print("\nТоп-5 фильмов в жанре комедия по среднему рейтингу:")
    print(top_comedy)

    top_2010 = preprocessor.get_top_by_year(year=2010, top_n=5, metric="weighted")
    print("\nТоп-5 фильмов в 2010 по среднему рейтингу:")
    print(top_2010)

    # Пример доступа к данным
    print("\nПример подготовленных данных:")
    print(prepared_data['movie_data'].head(3))
    print(f"\nРазреженная матрица: {prepared_data['sparse_user_item'].shape}")
