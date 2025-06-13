import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import warnings
import tkinter as tk
from tkinter import messagebox
import json
import os
import sys
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')


class MovieLensDataPreprocessor:
    def __init__(self, data_path='./', sample_fraction=None, dev_mode=False):
        """
        Инициализация с возможностью семплирования данных

        Parameters:
        - data_path: путь к папке с данными
        - sample_fraction: доля данных для семплирования (None для всех данных)
        """
        self.dev_mode = dev_mode
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
        if not self.dev_mode:
            return

        print(f"\n=== ДИАГНОСТИКА (режим разработчика) ===")
        print(f"Фильмов: {len(self.movies)}, Оценок: {len(self.ratings)}")
        print(f"Топ-5 фильмов по популярности:\n{self.ratings['movieId'].value_counts().head(5)}")

        # Распределение рейтингов (по частям для экономии памяти)
        plt.figure(figsize=(10, 5))
        self.ratings['rating'].hist(bins=10, edgecolor='black')
        plt.title('Распределение рейтингов фильмов')
        plt.xlabel('Рейтинг')
        plt.ylabel('Количество оценок')
        plt.show()

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

        valid_movies = set(self.movies['movieId']) & set(self.ratings['movieId'])
        self.ratings = self.ratings[self.ratings['movieId'].isin(valid_movies)]

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

        Parameters:
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

        Parameters:
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

    def get_top_by_genre(self, genre, top_n=10, min_ratings=50, metric="weighted", m_parameter=None):
        """
        Возвращает топ-N фильмов указанного жанра по выбранной метрике, используя существующие методы.

        Parameters:
        - genre: жанр (например, "Comedy", "Action")
        - top_n: количество фильмов в топе
        - min_ratings: минимальное количество оценок
        - metric: критерий сортировки ("weighted", "avg", "popularity")
        - m_parameter: параметр m для взвешенного рейтинга (если None - 90-й перцентиль)

        Returns:
        - DataFrame с колонками: movieId, title, year, rating_metric, rating_count, genres
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

        # 6. Добавляем год в результат (главное изменение)
        if 'year' not in result.columns:
            result['year'] = result['title'].str.extract(r'\((\d{4})\)')

        return result[["movieId", "title", "year", "rating_metric", "rating_count", "genres"]]

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


class CollaborativeFiltering:
    def __init__(self, data, k=20, metric='cosine'):
        self.data = data
        self.k = k
        self.metric = metric
        self.user_sim_matrix = None
        self.item_sim_matrix = None
        self.user_means = None
        self.item_means = None
        self.global_mean = None
        self.popular_items = None

    def fit(self, model_type='both'):
        """Обучение моделей с кэшированием популярных фильмов"""
        # Вычисляем средние оценки
        self.user_means = self.data['ratings'].groupby('userId')['rating'].mean().to_dict()
        self.item_means = self.data['ratings'].groupby('movieId')['rating'].mean().to_dict()
        self.global_mean = self.data['ratings']['rating'].mean()

        # Подготавливаем популярные фильмы (fallback)
        self._prepare_popular_items()

        if model_type in ['user', 'both']:
            self._fit_user_based()
        if model_type in ['item', 'both']:
            self._fit_item_based()

    def _fit_user_based(self):
        """Обучение User-Based модели"""
        print(f"{datetime.now()} - Обучение User-Based модели...")
        ratings_normalized = self.data['ratings'].copy()
        ratings_normalized['rating'] = ratings_normalized['rating'] - ratings_normalized['userId'].map(self.user_means)

        sparse_matrix = csr_matrix(
            (ratings_normalized['rating'].values,
             (ratings_normalized['userId'].map(self.data['user_mapping']),
              ratings_normalized['movieId'].map(self.data['movie_mapping']))),
            shape=(len(self.data['user_mapping']), len(self.data['movie_mapping']))
        )
        self.user_sim_matrix = cosine_similarity(sparse_matrix)

    from sklearn.neighbors import NearestNeighbors

    def _fit_item_based(self):
        """Приближенный Item-Based с использованием KNN"""
        print(f"{datetime.now()} - Обучение приближенной Item-Based модели...")

        # Нормализация оценок
        ratings_normalized = self.data['ratings'].copy()
        ratings_normalized['rating'] = ratings_normalized['rating'] - ratings_normalized['movieId'].map(self.item_means)

        # Создаем разреженную матрицу (items x users)
        sparse_matrix = csr_matrix(
            (ratings_normalized['rating'].values,
             (ratings_normalized['movieId'].map(self.data['movie_mapping']),
              ratings_normalized['userId'].map(self.data['user_mapping']))),
            shape=(len(self.data['movie_mapping']), len(self.data['user_mapping'])))

        # Используем приближенный метод поиска соседей
        model = NearestNeighbors(n_neighbors=self.k, metric='cosine', algorithm='brute', n_jobs=-1)
        model.fit(sparse_matrix)

        # Сохраняем только топ-K соседей для каждого элемента
        distances, indices = model.kneighbors(sparse_matrix)

        # Создаем разреженную матрицу схожести
        n_items = sparse_matrix.shape[0]
        rows = np.repeat(np.arange(n_items), self.k)
        cols = indices.flatten()
        data = (1 - distances.flatten())  # преобразуем расстояние в схожесть

        self.item_sim_matrix = csr_matrix((data, (rows, cols)), shape=(n_items, n_items))

    def _prepare_popular_items(self, n=100):
        """Подготовка списка популярных фильмов"""
        self.popular_items = (
            self.data['ratings']
            .groupby('movieId')
            .agg(rating_count=('rating', 'count'), avg_rating=('rating', 'mean'))
            .sort_values(['rating_count', 'avg_rating'], ascending=False)
            .head(n)
            .reset_index()
            .merge(self.data['movies'][['movieId', 'title', 'genres']], on='movieId')
        )
        self.popular_items['predicted_rating'] = self.popular_items['avg_rating']

    def recommend_for_new_user(self, survey_answers, n=10, model_type='hybrid'):
        """
        Рекомендации для нового пользователя на основе опроса

        Parameters:
        - survey_answers: словарь с ответами {movieId: rating}
        - n: количество рекомендаций
        - model_type: 'content', 'item', 'hybrid' (по умолчанию)

        Returns:
        - DataFrame с рекомендациями
        """
        # Проверка и очистка входных данных
        if not survey_answers or not isinstance(survey_answers, dict):
            return self._get_fallback_recommendations(n)

        # Фильтруем только существующие movieId
        valid_answers = {
            int(movie_id): float(rating)
            for movie_id, rating in survey_answers.items()
            if str(movie_id).isdigit()
               and int(movie_id) in self.data['movie_mapping']
               and 0.5 <= float(rating) <= 5.0
        }

        # Если нет валидных ответов, возвращаем популярные фильмы
        if not valid_answers:
            return self._get_fallback_recommendations(n)

        try:
            if model_type == 'content':
                return self._content_based_recommend(valid_answers, n)
            elif model_type == 'item':
                return self._safe_item_based_recommend(valid_answers, n)
            else:
                # Гибридный подход с обработкой ошибок
                try:
                    item_recs = self._safe_item_based_recommend(valid_answers, n * 2)
                except Exception as e:
                    print(f"Item-based failed: {str(e)}")
                    item_recs = self._get_fallback_recommendations(n)

                try:
                    content_recs = self._content_based_recommend(valid_answers, n * 2)
                except Exception as e:
                    print(f"Content-based failed: {str(e)}")
                    content_recs = self._get_fallback_recommendations(n)

                # Объединяем результаты
                hybrid = pd.concat([item_recs, content_recs]).drop_duplicates('movieId')
                return hybrid.head(n) if not hybrid.empty else self._get_fallback_recommendations(n)

        except Exception as e:
            print(f"Recommendation error: {str(e)}")
            return self._get_fallback_recommendations(n)

    def _safe_item_based_recommend(self, survey_answers, n):
        """Безопасная версия item-based рекомендаций"""
        try:
            return self._item_based_recommend_for_new_user(survey_answers, n)
        except IndexError:
            # Пересчитываем матрицу сходства при ошибке индексации
            self._fit_item_based()
            return self._item_based_recommend_for_new_user(survey_answers, n)

    def _get_fallback_recommendations(self, n):
        """Резервные рекомендации"""
        return self.popular_items.head(n)[['movieId', 'title', 'predicted_rating', 'genres']]

    def _item_based_recommend_for_new_user(self, survey_answers, n):
        """Item-Based рекомендации для нового пользователя"""
        rated_movies = list(survey_answers.keys())
        candidates = set(self.data['movie_mapping'].keys()) - set(rated_movies)

        pred_ratings = []
        for movie_id in candidates:
            # Проверяем, есть ли фильм в маппинге
            if movie_id not in self.data['movie_mapping']:
                continue  # Пропускаем неизвестные фильмы

            movie_idx = self.data['movie_mapping'][movie_id]

            sim_scores = []
            for rated_id, rating in survey_answers.items():
                rated_idx = self.data['movie_mapping'].get(rated_id)
                if rated_idx is None:
                    continue
                sim = self.item_sim_matrix[movie_idx][rated_idx]
                sim_scores.append((sim, rating))

            if sim_scores:
                sim_scores.sort(reverse=True)
                sim_scores = sim_scores[:self.k]
                weighted_sum = sum(s * r for s, r in sim_scores)
                sim_sum = sum(abs(s) for s, _ in sim_scores)
                pred_rating = weighted_sum / sim_sum if sim_sum > 0 else self.item_means.get(movie_id, self.global_mean)
            else:
                pred_rating = self.item_means.get(movie_id, self.global_mean)

            pred_ratings.append((movie_id, pred_rating))

        pred_ratings.sort(key=lambda x: x[1], reverse=True)
        result = pd.DataFrame(pred_ratings[:n], columns=['movieId', 'predicted_rating'])
        result = result.merge(self.data['movies'][['movieId', 'title', 'genres']], on='movieId')
        return result

    def _content_based_recommend(self, survey_answers, n):
        """Content-Based рекомендации на основе жанров понравившихся фильмов"""
        # Получаем жанры оцененных фильмов
        rated_movies = self.data['movies'][self.data['movies']['movieId'].isin(survey_answers.keys())]
        liked_genres = set()
        for genres in rated_movies['genres_list']:
            liked_genres.update(genres)

        # Ищем фильмы с похожими жанрами
        recommendations = self.data['movies'].copy()
        recommendations['genre_match'] = recommendations['genres_list'].apply(
            lambda x: len(set(x) & liked_genres) / len(liked_genres) if liked_genres else 0
        )

        # Исключаем уже оцененные фильмы
        recommendations = recommendations[~recommendations['movieId'].isin(survey_answers.keys())]

        # Сортируем по совпадению жанров и популярности
        top_movies = (
            recommendations.merge(
                self.popular_items[['movieId', 'rating_count']],
                on='movieId',
                how='left'
            )
            .sort_values(['genre_match', 'rating_count'], ascending=False)
            .head(n)
        )

        top_movies['predicted_rating'] = top_movies['rating_count']  # Просто для формата
        return top_movies[['movieId', 'title', 'predicted_rating', 'genres']]

    def recommend_for_existing_user(self, user_id, n=10, model_type='hybrid'):
        """Рекомендации для существующего пользователя"""
        if user_id not in self.data['user_mapping']:
            return self.popular_items.head(n)[['movieId', 'title', 'predicted_rating', 'genres']]

        if model_type == 'user':
            return self._user_based_recommend(user_id, n)
        elif model_type == 'item':
            return self._item_based_recommend(user_id, n)
        else:
            # Гибридный подход
            user_recs = self._user_based_recommend(user_id, n * 2)
            item_recs = self._item_based_recommend(user_id, n * 2)

            hybrid = pd.concat([user_recs, item_recs]).drop_duplicates('movieId')
            return hybrid.head(n)

    def _user_based_recommend(self, user_id, n):
        """User-Based рекомендации"""
        # Получаем индекс пользователя
        user_idx = self.data['user_mapping'].get(user_id)
        if user_idx is None:
            raise ValueError(f"Пользователь {user_id} не найден в данных")

        # Получаем оценки пользователя
        user_ratings = self.data['ratings'][self.data['ratings']['userId'] == user_id]

        # Находим k наиболее похожих пользователей
        sim_scores = list(enumerate(self.user_sim_matrix[user_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:self.k + 1]  # Исключаем самого пользователя

        # Собираем фильмы, которые пользователь еще не оценил
        all_movies = set(self.data['movie_mapping'].keys())
        rated_movies = set(user_ratings['movieId'])
        candidates = list(all_movies - rated_movies)

        # Предсказываем рейтинги для кандидатов
        pred_ratings = []
        for movie_id in candidates:
            movie_idx = self.data['movie_mapping'].get(movie_id)
            if movie_idx is None:
                continue

            # Собираем оценки похожих пользователей для этого фильма
            weighted_sum = 0
            sim_sum = 0
            count = 0

            for neighbor_idx, sim in sim_scores:
                neighbor_id = list(self.data['user_mapping'].keys())[
                    list(self.data['user_mapping'].values()).index(neighbor_idx)]

                # Проверяем, оценил ли сосед этот фильм
                neighbor_rating = self.data['ratings'][
                    (self.data['ratings']['userId'] == neighbor_id) &
                    (self.data['ratings']['movieId'] == movie_id)]

                if not neighbor_rating.empty:
                    weighted_sum += sim * neighbor_rating['rating'].values[0]
                    sim_sum += sim
                    count += 1

            if count > 0:
                pred_rating = weighted_sum / sim_sum
                pred_ratings.append((movie_id, pred_rating))

        # Сортируем по предсказанному рейтингу
        pred_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n = pred_ratings[:n]

        # Формируем результат
        result = pd.DataFrame(top_n, columns=['movieId', 'predicted_rating'])
        result = result.merge(self.data['movies'][['movieId', 'title']], on='movieId')

        return result[['movieId', 'title', 'predicted_rating']]

    def _item_based_recommend(self, user_id, n):
        """Item-Based рекомендации"""
        # Получаем оценки пользователя
        user_ratings = self.data['ratings'][self.data['ratings']['userId'] == user_id]
        if user_ratings.empty:
            raise ValueError(f"Пользователь {user_id} не оценил ни одного фильма")

        # Собираем фильмы, которые пользователь еще не оценил
        all_movies = set(self.data['movie_mapping'].keys())
        rated_movies = set(user_ratings['movieId'])
        candidates = list(all_movies - rated_movies)

        # Предсказываем рейтинги для кандидатов
        pred_ratings = []
        for movie_id in candidates:
            movie_idx = self.data['movie_mapping'].get(movie_id)
            if movie_idx is None:
                continue

            # Находим k наиболее похожих фильмов, которые пользователь оценил
            sim_scores = []
            for rated_movie_id in rated_movies:
                rated_movie_idx = self.data['movie_mapping'].get(rated_movie_id)
                if rated_movie_idx is None:
                    continue

                sim = self.item_sim_matrix[movie_idx][rated_movie_idx]
                rating = user_ratings[user_ratings['movieId'] == rated_movie_id]['rating'].values[0]
                sim_scores.append((sim, rating))

            # Берем топ-k похожих фильмов
            sim_scores.sort(reverse=True)
            sim_scores = sim_scores[:self.k]

            if len(sim_scores) == 0:
                continue

            # Взвешенное среднее
            weighted_sum = sum(s * r for s, r in sim_scores)
            sim_sum = sum(s for s, _ in sim_scores)

            pred_rating = weighted_sum / sim_sum
            pred_ratings.append((movie_id, pred_rating))

        # Сортируем по предсказанному рейтингу
        pred_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n = pred_ratings[:n]

        # Формируем результат
        result = pd.DataFrame(top_n, columns=['movieId', 'predicted_rating'])
        result = result.merge(self.data['movies'][['movieId', 'title']], on='movieId')

        return result[['movieId', 'title', 'predicted_rating']]

def run_dev_mode():
    print("=== Консольный режим проверки данных ===")

    # 1. Инициализация и загрузка данных
    print("\nЗагрузка данных...")
    preprocessor = MovieLensDataPreprocessor(
        data_path='ml-latest/',
        sample_fraction=0.1,
        dev_mode=True
    )

    # Запускаем весь пайплайн обработки
    prepared_data = preprocessor.run_pipeline()

    # 2. Вывод базовой статистики
    print("\n--- Основная статистика ---")
    print(f"Загружено фильмов: {len(preprocessor.movies)}")
    print(f"Загружено оценок: {len(preprocessor.ratings)}")
    print(f"Пример данных:\n{preprocessor.movies.head(2)}")

    # 3. Анализ топ-фильмов
    print("\n--- Топ фильмов ---")

    # По среднему рейтингу
    top_avg = preprocessor.get_top_by_avg_rating(top_n=5, min_ratings=50)
    print("\nТоп-5 по среднему рейтингу:")
    print(top_avg[['title', 'avg_rating', 'rating_count']].to_string(index=False))

    # По популярности
    top_popular = preprocessor.get_top_by_popularity(top_n=5, min_ratings=50)
    print("\nТоп-5 по популярности:")
    print(top_popular[['title', 'rating_count', 'avg_rating']].to_string(index=False))

    # По взвешенному рейтингу
    top_weighted = preprocessor.get_top_by_weighted_rating(top_n=5, min_ratings=50)
    print("\nТоп-5 по взвешенному рейтингу:")
    print(top_weighted[['title', 'weighted_rating', 'rating_count']].to_string(index=False))

    # 4. Примеры по жанрам и годам
    print("\n--- Примеры по категориям ---")

    # Для жанра "Comedy"
    top_comedy = preprocessor.get_top_by_genre(genre="Comedy", top_n=3, min_ratings=50)
    available_columns = top_comedy.columns.tolist()
    columns_to_show = ['title', 'rating_metric', 'genres']
    columns_to_show = [col for col in columns_to_show if col in available_columns]
    print("\nТоп-3 комедий:")
    print(top_comedy[['title', 'year', 'rating_metric']].to_string(index=False))

    # Для года 2010
    top_2010 = preprocessor.get_top_by_year(year=2010, top_n=3, min_ratings=50)
    print("\nТоп-3 фильмов 2010 года:")
    print(top_2010[['title', 'genres', 'rating_metric']].to_string(index=False))

    # 5. Тест рекомендательной системы
    print("\n--- Тест рекомендаций ---")

    # Инициализация модели
    cf_model = CollaborativeFiltering(prepared_data, k=20)

    # User-Based рекомендации
    print("\nUser-Based рекомендации для пользователя 1:")
    cf_model.fit(model_type='user')
    user_recs = cf_model.recommend_for_existing_user(user_id=1, n=3)
    print(user_recs[['title', 'predicted_rating']].to_string(index=False))

    # Item-Based рекомендации
    print("\nItem-Based рекомендации для пользователя 1:")
    cf_model.fit(model_type='item')
    item_recs = cf_model.recommend_for_existing_user(user_id=1, n=3)
    print(item_recs[['title', 'predicted_rating']].to_string(index=False))

    # 6. Информация о подготовленных данных
    print("\n--- Методанные ---")
    print(f"Размер user-item матрицы: {prepared_data['sparse_user_item'].shape}")
    print(f"Пример данных:\n{prepared_data['movie_data'].head(2)}")

    print("\nПроверка данных завершена!")

class MovieRecommendationApp:
    def __init__(self, root):
        self.root = root
        # Получаем размеры экрана
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Вычитаем 50 пикселей для панели задач (можно регулировать)
        window_height = screen_height - 50

        # Устанавливаем геометрию окна (без overrideredirect)
        self.root.geometry(f"{screen_width}x{window_height}+0+0")
        self.root.title("Добро пожаловать в Filmore!")
        self.root.configure(bg='pink2')

        # Окно будет поверх других окон (опционально)
        self.root.attributes('-topmost', True)
        self.root.after(1000, lambda: self.root.attributes('-topmost', False))

        # Стиль приложения (ваши оригинальные настройки)
        self.bg_color = 'pink2'
        self.frame_color = 'PaleVioletRed3'
        self.button_color = 'VioletRed3'
        self.text_color = 'old lace'
        self.font = ('Arial', 15)
        self.title_font = ('Arial', 20)

        # Инициализация компонентов (ваш оригинальный код)
        self.data_preprocessor = MovieLensDataPreprocessor(data_path='ml-latest/', sample_fraction=0.1, dev_mode=False)
        self.prepared_data = self.data_preprocessor.run_pipeline()
        self.cf_model = CollaborativeFiltering(self.prepared_data, k=20)
        self.cf_model.fit(model_type='both')

        self.user_profile = None
        self.current_user = None
        self.users_file = "users.json"
        self.users = {}

        self.load_users()
        self.create_main_menu()

    def load_users(self):
        """Загрузка данных пользователей из файла"""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r', encoding='utf-8') as f:
                    self.users = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                self.users = {}

    def save_users(self):
        """Сохранение данных пользователей в файл"""
        with open(self.users_file, 'w', encoding='utf-8') as f:
            json.dump(self.users, f, ensure_ascii=False, indent=4)

    def create_main_menu(self):
        """Главное меню приложения"""
        self.clear_window()

        # Фрейм для заголовка
        buttonframe1 = tk.Frame(self.root, bg=self.frame_color)
        buttonframe1.pack(padx=10, pady=40)

        tk.Label(buttonframe1, text="Добро пожаловать в Filmore!",
                 font=self.title_font, fg=self.text_color, bg=self.button_color
                 ).pack(padx=10, pady=10)

        # Пустой фрейм для отступа
        buttonframe2 = tk.Frame(self.root, bg=self.bg_color)
        buttonframe2.pack(padx=10, pady=20)
        tk.Label(buttonframe2, text="", font=self.title_font, bg=self.bg_color).pack(padx=10, pady=10)

        # Фрейм для кнопок
        buttonframe = tk.Frame(self.root, bg=self.frame_color)
        buttonframe.pack()

        # Кнопки меню
        tk.Button(buttonframe, text='Войти', font=self.font, height=2, width=14,
                  fg=self.text_color, bg=self.button_color, command=self.show_login
                  ).grid(row=0, column=0, padx=10, pady=10)

        tk.Button(buttonframe, text='Регистрация', font=self.font, height=2, width=14,
                  fg=self.text_color, bg=self.button_color, command=self.show_register
                  ).grid(row=1, column=0, padx=10, pady=10)

    def clear_window(self):
        """Очистка окна от всех виджетов"""
        for widget in self.root.winfo_children():
            widget.destroy()

    def show_login(self):
        """Окно входа"""
        login_window = tk.Toplevel(self.root)
        login_window.geometry("400x300")
        login_window.title("Вход в систему")
        login_window.configure(bg=self.bg_color)

        tk.Label(login_window, text="Логин:", font=self.font, bg=self.bg_color).pack(pady=10)
        self.login_entry = tk.Entry(login_window, font=self.font, width=20)
        self.login_entry.pack()

        tk.Label(login_window, text="Пароль:", font=self.font, bg=self.bg_color).pack(pady=10)
        self.password_entry = tk.Entry(login_window, font=self.font, width=20, show="*")
        self.password_entry.pack()

        tk.Button(login_window, text="Войти", font=self.font, fg=self.text_color,
                  bg=self.button_color, command=self.login).pack(pady=20)

    def login(self):
        """Авторизация пользователя"""
        login = self.login_entry.get()
        password = self.password_entry.get()

        if not login or not password:
            messagebox.showerror("Ошибка", "Введите логин и пароль", parent=self.root)
            return

        if login in self.users and self.users[login]['password'] == password:
            self.current_user = login
            self.user_profile = self.users[login].get('profile', {
                'favorite_genres': [],
                'favorite_years': [],
                'min_rating': 3.0,
                'watched_movies': {}
            })
            messagebox.showinfo("Успех", f"Добро пожаловать, {login}!", parent=self.root)
            self.login_entry.master.destroy()
            self.show_recommendation_interface()
        else:
            messagebox.showerror("Ошибка", "Неверный логин или пароль", parent=self.root)

    def show_register(self):
        """Окно регистрации"""
        register_window = tk.Toplevel(self.root)
        register_window.geometry("400x350")
        register_window.title("Регистрация")
        register_window.configure(bg=self.bg_color)

        tk.Label(register_window, text="Регистрация", font=self.title_font,
                 fg=self.text_color, bg=self.button_color).pack(pady=10)

        tk.Label(register_window, text="Логин:", font=self.font, bg=self.bg_color).pack()
        self.reg_login_entry = tk.Entry(register_window, font=self.font, width=20)
        self.reg_login_entry.pack()

        tk.Label(register_window, text="Пароль:", font=self.font, bg=self.bg_color).pack()
        self.reg_password_entry = tk.Entry(register_window, font=self.font, width=20, show="*")
        self.reg_password_entry.pack()

        tk.Label(register_window, text="Подтвердите пароль:", font=self.font, bg=self.bg_color).pack()
        self.reg_confirm_entry = tk.Entry(register_window, font=self.font, width=20, show="*")
        self.reg_confirm_entry.pack()

        tk.Button(register_window, text="Зарегистрироваться", font=self.font,
                  fg=self.text_color, bg=self.button_color, command=self.register).pack(pady=15)

    def register(self):
        """Регистрация нового пользователя"""
        login = self.reg_login_entry.get()
        password = self.reg_password_entry.get()
        confirm = self.reg_confirm_entry.get()

        if not login or not password:
            messagebox.showerror("Ошибка", "Введите логин и пароль", parent=self.root)
            return

        if password != confirm:
            messagebox.showerror("Ошибка", "Пароли не совпадают", parent=self.root)
            return

        if login in self.users:
            messagebox.showerror("Ошибка", "Пользователь с таким логином уже существует", parent=self.root)
            return

        self.users[login] = {
            'password': password,
            'profile': {
                'favorite_genres': [],
                'favorite_years': [],
                'min_rating': 3.0,
                'watched_movies': {}
            }
        }

        self.save_users()
        messagebox.showinfo("Успех", "Регистрация прошла успешно!", parent=self.root)
        self.reg_login_entry.master.destroy()
        self.current_user = login
        self.user_profile = self.users[login]['profile']
        self.show_recommendation_interface()

    def show_recommendation_interface(self):
        """Главный интерфейс после входа"""
        self.clear_window()

        buttonframe1 = tk.Frame(self.root, bg=self.frame_color)
        buttonframe1.pack(padx=10, pady=20)

        tk.Label(buttonframe1, text=f"Добро пожаловать, {self.current_user}!",
                 font=self.title_font, fg=self.text_color, bg=self.button_color
                 ).pack(padx=10, pady=10)

        buttonframe = tk.Frame(self.root, bg=self.frame_color)
        buttonframe.pack(pady=20)

        tk.Button(buttonframe, text="Пройти опрос", font=self.font, height=2, width=20,
                  fg=self.text_color, bg=self.button_color, command=self.run_user_survey
                  ).grid(row=0, column=0, padx=10, pady=10)

        tk.Button(buttonframe, text="Рекомендации", font=self.font, height=2, width=20,
                  fg=self.text_color, bg=self.button_color, command=self.show_recommendations
                  ).grid(row=0, column=1, padx=10, pady=10)

        tk.Button(buttonframe, text="Топ фильмов", font=self.font, height=2, width=20,
                  fg=self.text_color, bg=self.button_color, command=self.show_top_movies
                  ).grid(row=1, column=0, padx=10, pady=10)

        tk.Button(buttonframe, text="Выйти", font=self.font, height=2, width=20,
                  fg=self.text_color, bg=self.button_color, command=self.logout
                  ).grid(row=1, column=1, padx=10, pady=10)
    def show_top_movies(self):
        """Красивый интерфейс поиска топ фильмов"""
        self.clear_window()

        # Основной контейнер
        main_container = tk.Frame(self.root, bg=self.bg_color)
        main_container.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)

        # Заголовок
        tk.Label(main_container, text="Топ фильмов", font=self.title_font,
                 fg=self.text_color, bg=self.button_color).pack(pady=10)

        # Фрейм управления (3 строки)
        control_frame = tk.Frame(main_container, bg=self.bg_color)
        control_frame.pack(pady=5)

        # Строка 1: Критерий
        tk.Label(control_frame, text="Критерий:", font=self.font, bg=self.bg_color).grid(row=0, column=0, padx=5, pady=5,
                                                                                         sticky='e')
        criteria = ['По популярности', 'По жанру', 'По году']
        self.criterion_var = tk.StringVar(value=criteria[0])
        criterion_menu = tk.OptionMenu(control_frame, self.criterion_var, *criteria)
        criterion_menu.config(font=self.font, bg=self.button_color, fg=self.text_color)
        criterion_menu.grid(row=0, column=1, padx=5, sticky='w')

        # Строка 2: Количество фильмов
        tk.Label(control_frame, text="Кол-во:", font=self.font, bg=self.bg_color).grid(row=1, column=0, padx=5, pady=5,
                                                                                       sticky='e')
        self.top_n_entry = tk.Entry(control_frame, font=self.font, width=5)
        self.top_n_entry.insert(0, "5")
        self.top_n_entry.grid(row=1, column=1, padx=5, sticky='w')

        # Строка 3: Параметры поиска (жанр/год)
        self.params_frame = tk.Frame(control_frame, bg=self.bg_color)
        self.params_frame.grid(row=2, columnspan=2, pady=5)

        # Combobox для жанра
        self.genre_frame = tk.Frame(self.params_frame, bg=self.bg_color)
        tk.Label(self.genre_frame, text="Жанр:", font=self.font, bg=self.bg_color).pack(side=tk.LEFT)
        all_genres = sorted(set(np.concatenate(self.data_preprocessor.movies['genres_list'].values)))
        self.genre_var = tk.StringVar(value=all_genres[0] if all_genres else '')
        genre_menu = tk.OptionMenu(self.genre_frame, self.genre_var, *all_genres)
        genre_menu.config(font=self.font, bg=self.button_color, fg=self.text_color)
        genre_menu.pack(side=tk.LEFT, padx=5)

        # Поле для года
        self.year_frame = tk.Frame(self.params_frame, bg=self.bg_color)
        tk.Label(self.year_frame, text="Год:", font=self.font, bg=self.bg_color).pack(side=tk.LEFT)
        self.year_entry = tk.Entry(self.year_frame, font=self.font, width=6)
        self.year_entry.pack(side=tk.LEFT, padx=5)

        # Строка 4: Кнопка поиска (центрированная)
        btn_frame = tk.Frame(control_frame, bg=self.bg_color)
        btn_frame.grid(row=3, columnspan=2, pady=10)
        tk.Button(btn_frame, text="Найти", font=self.font, fg=self.text_color,
                  bg=self.button_color, command=self._perform_top_search).pack()

        # Поле результатов (компактное, выше кнопки Назад)
        result_frame = tk.Frame(main_container, bg=self.bg_color)
        result_frame.pack(pady=(10, 5), fill=tk.BOTH, expand=True)

        self.result_text = tk.Text(result_frame, font=self.font, height=8, width=50, wrap=tk.WORD)
        scrollbar = tk.Scrollbar(result_frame, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.result_text.config(state='disabled')

        # Кнопка назад (в самом низу)
        tk.Button(main_container, text="Назад", font=self.font, fg=self.text_color,
                  bg=self.button_color, command=self.show_recommendation_interface).pack(pady=10)

        # Первоначальная настройка
        self._update_top_movies_fields()
        self.criterion_var.trace('w', lambda *args: self._update_top_movies_fields())

    def _update_top_movies_fields(self):
        """Обновление видимости полей"""
        # Скрываем все параметры
        for widget in self.params_frame.winfo_children():
            widget.pack_forget()

        # Показываем нужные
        if self.criterion_var.get() == 'По жанру':
            self.genre_frame.pack(pady=5)
        elif self.criterion_var.get() == 'По году':
            self.year_frame.pack(pady=5)

    def _perform_top_search(self):
        """Выполнение поиска топ фильмов"""
        try:
            top_n = int(self.top_n_entry.get())
            if not 1 <= top_n <= 20:
                raise ValueError("Количество фильмов должно быть от 1 до 20")
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректное число от 1 до 20", parent=self.root)
            return

        try:
            criterion = self.criterion_var.get()

            if criterion == 'По популярности':
                results = self.data_preprocessor.get_top_by_popularity(top_n=top_n, min_ratings=50)
                results['rating_metric'] = results['rating_count']  # Добавляем недостающую колонку
                metric_name = "Количество оценок"

            elif criterion == 'По жанру':
                genre = self.genre_var.get()  # <-- Получаем значение жанра из виджета
                results = self.data_preprocessor.get_top_by_genre(
                    genre=genre,
                    top_n=top_n,
                    min_ratings=50,
                    metric="weighted"
                )
                metric_name = "Взвешенный рейтинг"

            elif criterion == 'По году':
                try:
                    year = int(self.year_entry.get())
                    if not 1900 <= year <= datetime.now().year:
                        raise ValueError(f"Год должен быть от 1900 до {datetime.now().year}")
                except ValueError:
                    messagebox.showerror("Ошибка",
                                         f"Введите корректный год (1900-{datetime.now().year})",
                                         parent=self.root)
                    return

                results = self.data_preprocessor.get_top_by_year(
                    year=year,
                    top_n=top_n,
                    min_ratings=50,
                    metric="weighted"
                )
                metric_name = "Взвешенный рейтинг"

            # Отображаем результаты
            self.result_text.config(state='normal')
            self.result_text.delete(1.0, tk.END)

            if results.empty:
                self.result_text.insert(tk.END, "Фильмы не найдены\n")
            else:
                for _, row in results.iterrows():
                    self.result_text.insert(tk.END, f"{row['title']}\n")
                    self.result_text.insert(tk.END, f"Жанры: {row['genres']}\n")
                    self.result_text.insert(tk.END, f"{metric_name}: {row['rating_metric']:.2f}\n")
                    if 'year' in row and not pd.isna(row['year']):
                        self.result_text.insert(tk.END, f"Год: {int(row['year'])}\n")
                    self.result_text.insert(tk.END, f"Оценок: {row['rating_count']}\n\n")

            self.result_text.config(state='disabled')
            self.result_text.see(tk.END)  # Прокручиваем к началу результатов

        except Exception as e:
            messagebox.showerror("Ошибка", str(e), parent=self.root)

    def logout(self):
        """Выход из системы"""
        self.save_profile()
        self.current_user = None
        self.user_profile = None
        self.create_main_menu()

    def save_profile(self):
        """Сохранение профиля пользователя"""
        if self.current_user and self.user_profile:
            self.users[self.current_user]['profile'] = self.user_profile
            self.save_users()

    def run_user_survey(self):
        """Опрос пользователя для создания профиля в текущем окне"""
        self.clear_window()

        tk.Label(self.root, text="Создание профиля", font=self.title_font,
                 fg=self.text_color, bg=self.button_color).pack(pady=10)

        # Жанры
        tk.Label(self.root, text="Выберите любимые жанры:", font=self.font, bg=self.bg_color).pack(pady=5)
        genres_frame = tk.Frame(self.root, bg=self.bg_color)
        genres_frame.pack(pady=5)

        all_genres = sorted(set(np.concatenate(self.data_preprocessor.movies['genres_list'].values)))
        self.genre_vars = []
        for i, genre in enumerate(all_genres):
            var = tk.BooleanVar()
            self.genre_vars.append((genre, var))
            tk.Checkbutton(genres_frame, text=genre, variable=var, font=self.font, bg=self.bg_color).grid(row=i // 4,
                                                                                                          column=i % 4,
                                                                                                          padx=5,
                                                                                                          pady=5)

        # Годы
        tk.Label(self.root, text="Введите диапазон годов (например, 1990-2020):", font=self.font,
                 bg=self.bg_color).pack(pady=5)
        years_frame = tk.Frame(self.root, bg=self.bg_color)
        years_frame.pack(pady=5)

        tk.Label(years_frame, text="От:", font=self.font, bg=self.bg_color).pack(side=tk.LEFT)
        self.year_from_entry = tk.Entry(years_frame, font=self.font, width=6)
        self.year_from_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(years_frame, text="До:", font=self.font, bg=self.bg_color).pack(side=tk.LEFT)
        self.year_to_entry = tk.Entry(years_frame, font=self.font, width=6)
        self.year_to_entry.pack(side=tk.LEFT, padx=5)

        # Минимальный рейтинг
        tk.Label(self.root, text="Минимальный рейтинг (0.5-5.0):", font=self.font, bg=self.bg_color).pack(pady=5)
        self.min_rating_entry = tk.Entry(self.root, font=self.font, width=10)
        self.min_rating_entry.insert(0, "3.0")
        self.min_rating_entry.pack(pady=5)

        # Оценка фильмов
        tk.Label(self.root, text="Оцените фильмы (1-5):", font=self.font, bg=self.bg_color).pack(pady=5)
        movies_frame = tk.Frame(self.root, bg=self.bg_color)
        movies_frame.pack(pady=5)

        top_movies = self.data_preprocessor.get_top_by_popularity(top_n=5, min_ratings=50)
        self.movie_ratings = {}
        for _, row in top_movies.iterrows():
            movie_id = row['movieId']
            title = row['title']
            tk.Label(movies_frame, text=title, font=self.font, bg=self.bg_color).pack(anchor='w')
            rating_entry = tk.Entry(movies_frame, font=self.font, width=5)
            rating_entry.pack(anchor='w')
            self.movie_ratings[movie_id] = rating_entry

        def save_survey():
            # Сохраняем жанры
            self.user_profile['favorite_genres'] = [genre for genre, var in self.genre_vars if var.get()]

            # Сохраняем годы
            try:
                year_from = int(self.year_from_entry.get()) if self.year_from_entry.get() else 1900
                year_to = int(self.year_to_entry.get()) if self.year_to_entry.get() else 2025
                self.user_profile['favorite_years'] = list(range(year_from, year_to + 1))
            except ValueError:
                messagebox.showerror("Ошибка", "Введите корректные годы", parent=self.root)
                return

            # Сохраняем минимальный рейтинг
            try:
                min_rating = float(self.min_rating_entry.get())
                if 0.5 <= min_rating <= 5.0:
                    self.user_profile['min_rating'] = min_rating
                else:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Ошибка", "Введите рейтинг от 0.5 до 5.0", parent=self.root)
                return

            # Сохраняем оценки фильмов
            for movie_id, entry in self.movie_ratings.items():
                rating = entry.get()
                if rating:
                    try:
                        rating = float(rating)
                        if 1.0 <= rating <= 5.0:
                            self.user_profile['watched_movies'][str(movie_id)] = rating
                    except ValueError:
                        pass

            self.save_profile()
            messagebox.showinfo("Успех", "Профиль сохранен!", parent=self.root)
            self.show_recommendation_interface()

        tk.Button(self.root, text="Сохранить", font=self.font, fg=self.text_color,
                  bg=self.button_color, command=save_survey).pack(pady=10)
        tk.Button(self.root, text="Назад", font=self.font, fg=self.text_color,
                  bg=self.button_color, command=self.show_recommendation_interface).pack(pady=10)

    def show_recommendations(self):
        """Показать рекомендации в текущем окне"""
        if not self.user_profile or not (
                self.user_profile.get('favorite_genres') or self.user_profile.get('watched_movies')):
            messagebox.showwarning("Предупреждение", "Сначала пройдите опрос", parent=self.root)
            return

        self.clear_window()

        tk.Label(self.root, text="Рекомендации для вас", font=self.title_font,
                 fg=self.text_color, bg=self.button_color).pack(pady=10)

        # Получаем рекомендации
        survey_answers = {int(k): v for k, v in self.user_profile['watched_movies'].items()}
        recommendations = self.cf_model.recommend_for_new_user(survey_answers, n=10, model_type='hybrid')

        # Отображаем рекомендации
        text_area = tk.Text(self.root, font=self.font, height=15, width=50)
        text_area.pack(pady=10)
        for _, row in recommendations.iterrows():
            text_area.insert(tk.END, f"{row['title']} ({row['genres']})\n")
            text_area.insert(tk.END, f"Предсказанный рейтинг: {row['predicted_rating']:.2f}\n\n")
        text_area.config(state='disabled')

        tk.Button(self.root, text="Назад", font=self.font, fg=self.text_color,
                  bg=self.button_color, command=self.show_recommendation_interface).pack(pady=10)


if __name__ == "__main__":
    dev_mode = len(sys.argv) > 1 and sys.argv[1] == "--dev"
    if dev_mode:
        run_dev_mode()
    else:
        # Запускаем GUI по умолчанию
        root = tk.Tk()
        app = MovieRecommendationApp(root)
        root.mainloop()