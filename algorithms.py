import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
import pandas as pd

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
        self.user_means = self.data['ratings'].groupby('userId')['rating'].mean().to_dict()
        self.item_means = self.data['ratings'].groupby('movieId')['rating'].mean().to_dict()
        self.global_mean = self.data['ratings']['rating'].mean()
        self._prepare_popular_items()

        if model_type in ['user', 'both']:
            self._fit_user_based()
        if model_type in ['item', 'both']:
            self._fit_item_based()

    def _fit_user_based(self):
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

    def _fit_item_based(self):
        print(f"{datetime.now()} - Обучение приближенной Item-Based модели...")
        ratings_normalized = self.data['ratings'].copy()
        ratings_normalized['rating'] = ratings_normalized['rating'] - ratings_normalized['movieId'].map(self.item_means)

        rows = ratings_normalized['movieId'].map(self.data['movie_mapping'])
        cols = ratings_normalized['userId'].map(self.data['user_mapping'])

        if (rows < 0).any() or (cols < 0).any():
            invalid_movies = ratings_normalized[rows < 0]['movieId'].unique()
            invalid_users = ratings_normalized[cols < 0]['userId'].unique()
            error_msg = f"Обнаружены отрицательные индексы!\nФильмы: {invalid_movies}\nПользователи: {invalid_users}"
            raise ValueError(error_msg)

        sparse_matrix = csr_matrix(
            (ratings_normalized['rating'].values, (rows, cols)),
            shape=(len(self.data['movie_mapping']), len(self.data['user_mapping']))
        )

        if sparse_matrix.shape[0] == 0 or sparse_matrix.shape[1] == 0:
            raise ValueError("Размерность матрицы не может быть нулевой")

        model = NearestNeighbors(n_neighbors=self.k, metric='cosine', algorithm='brute', n_jobs=-1)
        model.fit(sparse_matrix)
        distances, indices = model.kneighbors(sparse_matrix)

        n_items = sparse_matrix.shape[0]
        rows = np.repeat(np.arange(n_items), self.k)
        cols = indices.flatten()
        data = (1 - distances.flatten())

        self.item_sim_matrix = csr_matrix((data, (rows, cols)), shape=(n_items, n_items))

    def _prepare_popular_items(self, n=100):
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

    def recommend_for_new_user(self, survey_answers, n=10, model_type='hybrid', user_profile=None):
        self.user_profile = user_profile or {}
        valid_answers = {
            int(movie_id): float(rating)
            for movie_id, rating in (survey_answers or {}).items()
            if str(movie_id).isdigit()
               and int(movie_id) in self.data['movie_mapping']
               and 0.5 <= float(rating) <= 5.0
        }

        if model_type == 'content' or not valid_answers:
            return self._content_based_recommend(valid_answers, n, user_profile)

        try:
            content_recs = self._content_based_recommend(valid_answers, n * 2, user_profile)
            item_recs = self._safe_item_based_recommend(valid_answers, n * 2)
            hybrid = pd.concat([content_recs, item_recs]).drop_duplicates('movieId')
            return hybrid.head(n) if not hybrid.empty else self._get_fallback_recommendations(n, user_profile)
        except Exception as e:
            print(f"Recommendation error: {str(e)}")
            return self._get_fallback_recommendations(n, user_profile)

    def _safe_item_based_recommend(self, survey_answers, n):
        try:
            return self._item_based_recommend_for_new_user(survey_answers, n)
        except IndexError:
            self._fit_item_based()
            return self._item_based_recommend_for_new_user(survey_answers, n)

    def _get_fallback_recommendations(self, n, user_profile=None):
        popular = self.popular_items.copy()
        if user_profile and user_profile.get('favorite_genres'):
            popular = popular[
                popular['genres_list'].apply(
                    lambda x: any(g in x for g in user_profile['favorite_genres'])
                )
            ]
        return popular.head(n)[['movieId', 'title', 'genres']]

    def _item_based_recommend_for_new_user(self, survey_answers, n):
        if not survey_answers or not isinstance(survey_answers, dict):
            print("Ошибка: survey_answers должен быть непустым словарем")
            return self._get_fallback_recommendations(n)

        if not hasattr(self, 'item_sim_matrix') or self.item_sim_matrix is None:
            print("Ошибка: item_sim_matrix не инициализирована")
            return self._get_fallback_recommendations(n)

        if 'movie_mapping' not in self.data or not self.data['movie_mapping']:
            print("Ошибка: отсутствует movie_mapping")
            return self._get_fallback_recommendations(n)

        try:
            rated_movies = []
            for movie_id in survey_answers.keys():
                try:
                    rated_movies.append(int(movie_id))
                except (ValueError, TypeError):
                    print(f"Пропущен некорректный movie_id: {movie_id}")
                    continue

            valid_movies = set(self.data['movie_mapping'].keys())
            candidates = list(valid_movies - set(rated_movies))

            if not candidates:
                print("Нет кандидатов для рекомендаций")
                return self._get_fallback_recommendations(n)

            movie_scores = []
            matrix_shape = self.item_sim_matrix.shape

            for movie_id in candidates:
                movie_idx = self.data['movie_mapping'].get(movie_id, -1)
                if movie_idx < 0 or movie_idx >= matrix_shape[0]:
                    continue

                total_sim = 0.0
                valid_rated_count = 0

                for rated_id, rating in survey_answers.items():
                    try:
                        rated_id_int = int(rated_id)
                        rated_idx = self.data['movie_mapping'].get(rated_id_int, -1)
                        if rated_idx < 0 or rated_idx >= matrix_shape[1]:
                            continue

                        sim = self.item_sim_matrix[movie_idx, rated_idx]
                        if not np.isnan(sim):
                            total_sim += sim
                            valid_rated_count += 1
                    except Exception:
                        continue

                if valid_rated_count > 0:
                    movie_scores.append((movie_id, total_sim))

            movie_scores.sort(key=lambda x: x[1], reverse=True)
            top_movies = [movie_id for movie_id, score in movie_scores[:n]]

            result = pd.DataFrame(top_movies, columns=['movieId'])
            result = result.merge(
                self.data['movies'][['movieId', 'title', 'genres']],
                on='movieId',
                how='left'
            ).dropna(subset=['title'])

            return result if not result.empty else self._get_fallback_recommendations(n)

        except Exception as e:
            print(f"Критическая ошибка в рекомендательной системе: {str(e)}")
            return self._get_fallback_recommendations(n)

    def _content_based_recommend(self, survey_answers, n, user_profile=None):
        if not user_profile:
            return self._get_fallback_recommendations(n)

        liked_genres = set(user_profile.get('favorite_genres', []))

        if survey_answers:
            rated_movies = self.data['movies'][self.data['movies']['movieId'].isin(survey_answers.keys())]
            for genres in rated_movies['genres_list']:
                liked_genres.update(genres)

        recs = self.data['movies'].copy()
        recs = recs[recs['genres_list'].apply(
            lambda x: all(genre in liked_genres for genre in x)
        )]

        if user_profile.get('favorite_years'):
            recs = recs[recs['year'].isin(user_profile['favorite_years'])]

        recs = recs[~recs['movieId'].isin(survey_answers.keys())]
        recs = recs.merge(
            self.popular_items[['movieId', 'rating_count']],
            on='movieId',
            how='left'
        ).sort_values('rating_count', ascending=False).head(n)

        return recs[['movieId', 'title', 'genres']]

    def recommend_for_existing_user(self, user_id, n=10, model_type='hybrid'):
        if user_id not in self.data['user_mapping']:
            return self._get_fallback_recommendations(n)

        if model_type in ['user', 'hybrid']:
            user_recs = self._user_based_recommend(user_id, n)

        if model_type in ['item', 'hybrid']:
            item_recs = self._item_based_recommend(user_id, n)

        if model_type == 'hybrid':
            recs = pd.concat([user_recs, item_recs]).drop_duplicates('movieId')
        elif model_type == 'user':
            recs = user_recs
        else:
            recs = item_recs

        return recs.head(n) if not recs.empty else self._get_fallback_recommendations(n)

    def _user_based_recommend(self, user_id, n):
        user_idx = self.data['user_mapping'].get(user_id)
        if user_idx is None:
            raise ValueError(f"Пользователь {user_id} не найден в данных")

        user_ratings = self.data['ratings'][self.data['ratings']['userId'] == user_id]
        sim_scores = list(enumerate(self.user_sim_matrix[user_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:self.k + 1]

        all_movies = set(self.data['movie_mapping'].keys())
        rated_movies = set(user_ratings['movieId'])
        candidates = list(all_movies - rated_movies)

        pred_ratings = []
        for movie_id in candidates:
            movie_idx = self.data['movie_mapping'].get(movie_id)
            if movie_idx is None:
                continue

            weighted_sum = 0
            sim_sum = 0
            count = 0

            for neighbor_idx, sim in sim_scores:
                neighbor_id = list(self.data['user_mapping'].keys())[
                    list(self.data['user_mapping'].values()).index(neighbor_idx)]

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

        pred_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n = pred_ratings[:n]

        result = pd.DataFrame(top_n, columns=['movieId', 'predicted_rating'])
        result = result.merge(self.data['movies'][['movieId', 'title']], on='movieId')
        return result[['movieId', 'title', 'predicted_rating']]

    def _item_based_recommend(self, user_id, n):
        user_ratings = self.data['ratings'][self.data['ratings']['userId'] == user_id]
        if user_ratings.empty:
            raise ValueError(f"Пользователь {user_id} не оценил ни одного фильма")

        all_movies = set(self.data['movie_mapping'].keys())
        rated_movies = set(user_ratings['movieId'])
        candidates = list(all_movies - rated_movies)

        pred_ratings = []
        for movie_id in candidates:
            movie_idx = self.data['movie_mapping'].get(movie_id)
            if movie_idx is None:
                continue

            sim_scores = []
            for rated_movie_id in rated_movies:
                rated_movie_idx = self.data['movie_mapping'].get(rated_movie_id)
                if rated_movie_idx is None:
                    continue

                sim = self.item_sim_matrix[movie_idx, rated_movie_idx]
                rating = user_ratings[user_ratings['movieId'] == rated_movie_id]['rating'].values[0]
                sim_scores.append((sim, rating))

            sim_scores.sort(reverse=True, key=lambda x: x[0])
            sim_scores = sim_scores[:self.k]

            if not sim_scores:
                pred_rating = self.item_means.get(movie_id, self.global_mean)
            else:
                weighted_sum = sum(s * r for s, r in sim_scores)
                sim_sum = sum(abs(s) for s, _ in sim_scores)

                if sim_sum < 1e-6:
                    pred_rating = self.item_means.get(movie_id, self.global_mean)
                else:
                    pred_rating = weighted_sum / sim_sum

            pred_ratings.append((movie_id, pred_rating))

        pred_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n = pred_ratings[:n]

        result = pd.DataFrame(top_n, columns=['movieId', 'predicted_rating'])
        result = result.merge(
            self.data['movies'][['movieId', 'title', 'genres']],
            on='movieId',
            how='left'
        )
        return result[['movieId', 'title', 'predicted_rating', 'genres']]
