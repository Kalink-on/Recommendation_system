from data_processing import MovieLensDataPreprocessor


def run_dev_mode():
    print("=== Консольный режим проверки данных ===")
    print("\nЗагрузка данных...")
    preprocessor = MovieLensDataPreprocessor(
        data_path='ml-latest/',
        sample_fraction=0.1,
        dev_mode=True
    )
    prepared_data = preprocessor.run_pipeline()

    print("\n--- Основная статистика ---")
    print(f"Загружено фильмов: {len(preprocessor.movies)}")
    print(f"Загружено оценок: {len(preprocessor.ratings)}")
    print(f"Пример данных:\n{preprocessor.movies.head(2)}")

    print("\n--- Топ фильмов ---")
    top_avg = preprocessor.get_top_by_avg_rating(top_n=5, min_ratings=50)
    print("\nТоп-5 по среднему рейтингу:")
    print(top_avg[['title', 'avg_rating', 'rating_count']].to_string(index=False))

    top_popular = preprocessor.get_top_by_popularity(top_n=5, min_ratings=50)
    print("\nТоп-5 по популярности:")
    print(top_popular[['title', 'rating_count', 'avg_rating']].to_string(index=False))

    top_weighted = preprocessor.get_top_by_weighted_rating(top_n=5, min_ratings=50)
    print("\nТоп-5 по взвешенному рейтингу:")
    print(top_weighted[['title', 'weighted_rating', 'rating_count']].to_string(index=False))

    print("\n--- Примеры по категориям ---")
    top_comedy = preprocessor.get_top_by_genre(genre="Comedy", top_n=3, min_ratings=50)
    available_columns = top_comedy.columns.tolist()
    columns_to_show = ['title', 'rating_metric', 'genres']
    columns_to_show = [col for col in columns_to_show if col in available_columns]
    print("\nТоп-3 комедий:")
    print(top_comedy[['title', 'year', 'rating_metric']].to_string(index=False))

    top_2010 = preprocessor.get_top_by_year(year=2010, top_n=3, min_ratings=50)
    print("\nТоп-3 фильмов 2010 года:")
    print(top_2010[['title', 'genres', 'rating_metric']].to_string(index=False))

    print("\n--- Методанные ---")
    print(f"Размер user-item матрицы: {prepared_data['sparse_user_item'].shape}")
    print(f"Пример данных:\n{prepared_data['movie_data'].head(2)}")

    print("\nПроверка данных завершена!")

if __name__ == "__main__":
    run_dev_mode()
