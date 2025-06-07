import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import warnings
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

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
        if not survey_answers:
            return self.popular_items.head(n)[['movieId', 'title', 'predicted_rating', 'genres']]

        if model_type == 'content':
            return self._content_based_recommend(survey_answers, n)
        elif model_type == 'item':
            return self._item_based_recommend_for_new_user(survey_answers, n)
        else:
            # Гибридный подход
            item_recs = self._item_based_recommend_for_new_user(survey_answers, n * 2)
            content_recs = self._content_based_recommend(survey_answers, n * 2)

            # Объединяем и удаляем дубликаты
            hybrid = pd.concat([item_recs, content_recs]).drop_duplicates('movieId')
            return hybrid.head(n)

    def _item_based_recommend_for_new_user(self, survey_answers, n):
        """Item-Based рекомендации для нового пользователя"""
        rated_movies = list(survey_answers.keys())
        candidates = set(self.data['movie_mapping'].keys()) - set(rated_movies)

        pred_ratings = []
        for movie_id in candidates:
            movie_idx = self.data['movie_mapping'].get(movie_id)
            if movie_idx is None:
                continue

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


class MovieRecommenderSystem:
    def __init__(self, data_preprocessor):
        self.data = data_preprocessor
        self.cf_model = None
        self.user_profile = None

        # Проверка загрузки данных в конструкторе
        if self.data.ratings is None:
            print("\nИнициализация данных...")
            self.data.load_data()
            self.data.clean_data()
            print("Данные успешно загружены и обработаны")

    def initialize_models(self):
        """Инициализация моделей рекомендаций"""
        print("\nИнициализация моделей рекомендаций...")
        prepared_data = self.data.prepare_recommendation_data()
        self.cf_model = CollaborativeFiltering(prepared_data, k=20)
        self.cf_model.fit(model_type='both')
        print("Модели инициализированы и обучены.")

    def run_user_survey(self):
        """Проведение опроса пользователя для создания профиля"""
        print("\n=== Анкета для создания профиля пользователя ===")
        self.user_profile = {
            'favorite_genres': [],
            'favorite_years': [],
            'min_rating': 3.0,
            'watched_movies': {}
        }

        # Выбор любимых жанров
        all_genres = sorted(set(np.concatenate(self.data.movies['genres_list'].values)))
        print("\nДоступные жанры:")
        for i, genre in enumerate(all_genres, 1):
            print(f"{i}. {genre}")

        selected = input("\nВведите номера любимых жанров через запятую (например: 1,3,5): ")
        selected_indices = [int(i.strip()) - 1 for i in selected.split(',') if i.strip().isdigit()]
        self.user_profile['favorite_genres'] = [all_genres[i] for i in selected_indices if 0 <= i < len(all_genres)]

        # Выбор предпочитаемых годов выпуска
        min_year = int(self.data.movies['year'].min())
        max_year = int(self.data.movies['year'].max())
        print(f"\nГоды выпуска фильмов: от {min_year} до {max_year}")
        selected = input("Введите предпочитаемые годы или диапазоны через запятую (например: 1990-2000, 2015): ")

        years = []
        for part in selected.split(','):
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                years.extend(range(start, end + 1))
            elif part.isdigit():
                years.append(int(part))
        self.user_profile['favorite_years'] = [y for y in years if min_year <= y <= max_year]

        # Минимальный рейтинг
        min_r = input("\nМинимальный средний рейтинг фильмов (по шкале 0.5-5.0, по умолчанию 3.0): ")
        if min_r.replace('.', '', 1).isdigit():
            self.user_profile['min_rating'] = max(0.5, min(5.0, float(min_r)))

        # Оценка некоторых фильмов (опционально)
        print("\nХотите оценить некоторые фильмы для улучшения рекомендаций? (y/n)")
        if input().lower() == 'y':
            self.rate_movies_interactive()

        print("\nСпасибо! Ваш профиль сохранен.")

    def rate_movies_interactive(self):
        """Интерактивная оценка фильмов пользователем"""
        print("\nОцените несколько фильмов (1-5 звезд) или нажмите Enter для пропуска.")

        # Показываем топ фильмов по популярности для оценки
        top_movies = self.data.get_top_by_popularity(top_n=20, min_ratings=50)

        for _, row in top_movies.iterrows():
            print(f"\nФильм: {row['title']} ({row['genres']})")
            print(f"Средний рейтинг: {row['avg_rating']:.1f}, оценок: {row['rating_count']}")
            rating = input("Ваша оценка (1-5) или Enter для пропуска: ")

            if rating.replace('.', '', 1).isdigit():
                rating = max(0.5, min(5.0, float(rating)))
                self.user_profile['watched_movies'][row['movieId']] = rating

    def get_recommendations_based_on_profile(self, n=10):
        """Получение рекомендаций на основе профиля пользователя"""
        if not self.user_profile:
            print("Профиль пользователя не создан. Запустите опрос сначала.")
            return None

        # 1. Фильтрация по жанрам и годам
        filtered = self.data.movies.copy()

        if self.user_profile['favorite_genres']:
            genre_mask = filtered['genres'].apply(
                lambda g: any(genre in g for genre in self.user_profile['favorite_genres'])
            )
            filtered = filtered[genre_mask]

        if self.user_profile['favorite_years']:
            filtered = filtered[filtered['year'].isin(self.user_profile['favorite_years'])]

        # 2. Добавляем информацию о рейтингах
        movie_stats = (
            self.data.ratings.groupby('movieId')
            .agg(avg_rating=('rating', 'mean'), rating_count=('rating', 'count'))
            .reset_index()
        )
        filtered = filtered.merge(movie_stats, on='movieId', how='left')

        # 3. Применяем минимальный рейтинг
        filtered = filtered[filtered['avg_rating'] >= self.user_profile['min_rating']]

        # 4. Исключаем уже оцененные фильмы
        if self.user_profile['watched_movies']:
            filtered = filtered[~filtered['movieId'].isin(self.user_profile['watched_movies'].keys())]

        # 5. Сортируем по взвешенному рейтингу
        C = filtered['avg_rating'].mean()
        m = filtered['rating_count'].quantile(0.75)
        filtered['weighted_rating'] = ((filtered['avg_rating'] * filtered['rating_count']) + (C * m)) / (
                    filtered['rating_count'] + m)

        recommendations = filtered.sort_values('weighted_rating', ascending=False).head(n)

        return recommendations[['movieId', 'title', 'year', 'genres', 'avg_rating', 'rating_count']]

    def interactive_movie_search(self):
        """Интерактивный поиск фильмов с фильтрами"""
        print("\n=== Расширенный поиск фильмов ===")

        # Инициализация фильтров
        filters = {
            'title': '',
            'genres': [],
            'year_from': None,
            'year_to': None,
            'min_rating': None,
            'min_votes': None
        }

        while True:
            print("\nТекущие фильтры:")
            print(f"1. Название: {filters['title'] or 'не задано'}")
            print(f"2. Жанры: {', '.join(filters['genres']) or 'не заданы'}")
            print(f"3. Годы: {filters['year_from'] or '?'} - {filters['year_to'] or '?'}")
            print(f"4. Минимальный рейтинг: {filters['min_rating'] or 'не задан'}")
            print(f"5. Минимальное количество оценок: {filters['min_votes'] or 'не задано'}")
            print("\nВыберите параметр для изменения (1-5) или Enter для поиска:")

            choice = input()
            if not choice:
                break

            if choice == '1':
                filters['title'] = input("Введите часть названия фильма: ").strip()
            elif choice == '2':
                all_genres = sorted(set(np.concatenate(self.data.movies['genres_list'].values)))
                print("\nДоступные жанры:")
                for i, genre in enumerate(all_genres, 1):
                    print(f"{i}. {genre}")
                selected = input("\nВведите номера жанров через запятую: ")
                selected_indices = [int(i.strip()) - 1 for i in selected.split(',') if i.strip().isdigit()]
                filters['genres'] = [all_genres[i] for i in selected_indices if 0 <= i < len(all_genres)]
            elif choice == '3':
                min_year = int(self.data.movies['year'].min())
                max_year = int(self.data.movies['year'].max())
                print(f"Доступные годы: {min_year}-{max_year}")
                filters['year_from'] = int(input("Год от: ") or min_year)
                filters['year_to'] = int(input("Год до: ") or max_year)
            elif choice == '4':
                filters['min_rating'] = float(input("Минимальный средний рейтинг (0.5-5.0): ") or 0)
            elif choice == '5':
                filters['min_votes'] = int(input("Минимальное количество оценок: ") or 0)

        # Применяем фильтры
        results = self.data.movies.copy()

        # Фильтр по названию
        if filters['title']:
            results = results[results['title'].str.contains(filters['title'], case=False, na=False)]

        # Фильтр по жанрам
        if filters['genres']:
            genre_mask = results['genres'].apply(
                lambda g: all(genre in g for genre in filters['genres'])
            )
            results = results[genre_mask]

        # Фильтр по годам
        if filters['year_from'] or filters['year_to']:
            year_from = filters['year_from'] or self.data.movies['year'].min()
            year_to = filters['year_to'] or self.data.movies['year'].max()
            results = results[(results['year'] >= year_from) & (results['year'] <= year_to)]

        # Добавляем информацию о рейтингах
        movie_stats = (
            self.data.ratings.groupby('movieId')
            .agg(avg_rating=('rating', 'mean'), rating_count=('rating', 'count'))
            .reset_index()
        )
        results = results.merge(movie_stats, on='movieId', how='left')

        # Фильтр по рейтингу
        if filters['min_rating']:
            results = results[results['avg_rating'] >= filters['min_rating']]

        # Фильтр по количеству оценок
        if filters['min_votes']:
            results = results[results['rating_count'] >= filters['min_votes']]

        # Сортируем результаты
        if not results.empty:
            results = results.sort_values('avg_rating', ascending=False)
            print(f"\nНайдено {len(results)} фильмов. Топ-10:")
            return results.head(10)[['movieId', 'title', 'year', 'genres', 'avg_rating', 'rating_count']]
        else:
            print("\nПо вашему запросу ничего не найдено.")
            return None

    def run_interactive_loop(self):
        """Основной интерактивный цикл программы"""
        print("\n=== Система рекомендаций фильмов ===")

        # Инициализация моделей
        self.initialize_models()

        while True:
            print("\nГлавное меню:")
            print("1. Пройти опрос для создания профиля")
            print("2. Получить рекомендации на основе профиля")
            print("3. Расширенный поиск фильмов")
            print("4. Получить рекомендации для существующего пользователя")
            print("5. Выход")

            choice = input("Выберите действие (1-5): ")

            if choice == '1':
                self.run_user_survey()
            elif choice == '2':
                if not self.user_profile:
                    print("Сначала создайте профиль через опрос.")
                    continue
                recs = self.get_recommendations_based_on_profile(10)
                if recs is not None:
                    print("\nРекомендуемые для вас фильмы:")
                    print(recs.to_string(index=False))
            elif choice == '3':
                results = self.interactive_movie_search()
                if results is not None:
                    print(results.to_string(index=False))
            elif choice == '4':
                user_id = int(input("Введите ID пользователя (1-610): ") or 1)
                recs = self.cf_model.recommend_for_existing_user(user_id, 10)
                print("\nРекомендации для пользователя:")
                print(recs.to_string(index=False))
            elif choice == '5':
                print("До свидания!")
                break


if __name__ == "__main__":
    # Инициализация системы
    print("Запуск системы рекомендаций фильмов...")

    # Создаем препроцессор с меньшим набором данных для демонстрации
    preprocessor = MovieLensDataPreprocessor(
        data_path='ml-latest/',
        sample_fraction=0.1  # Используем 10% данных для демонстрации
    )

    # Запускаем интерактивный интерфейс
    recommender = MovieRecommenderSystem(preprocessor)
    recommender.run_interactive_loop()

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

    # Создаем и обучаем модель коллаборативной фильтрации
    cf = CollaborativeFiltering(prepared_data, k=20)

    # User-Based рекомендации
    print("\nОбучение User-Based модели...")
    cf.fit(model_type='user')

    # Получаем рекомендации для пользователя 1
    user_id = 1
    print(f"\nUser-Based рекомендации для пользователя {user_id}:")
    user_recs = cf.recommend_for_existing_user(user_id, n=5, model_type='user')
    print(user_recs)

    # Item-Based рекомендации
    print("\nОбучение Item-Based модели...")
    cf.fit(model_type='item')

    # Получаем рекомендации для пользователя 1
    print(f"\nItem-Based рекомендации для пользователя {user_id}:")
    item_recs = cf.recommend_for_existing_user(user_id, n=5, model_type='item')
    print(item_recs)

    # Пример доступа к данным
    print("\nПример подготовленных данных:")
    print(prepared_data['movie_data'].head(3))
    print(f"\nРазреженная матрица: {prepared_data['sparse_user_item'].shape}")
