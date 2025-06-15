import tkinter as tk
from tkinter import messagebox
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime


class MovieRecommendationApp:
    def __init__(self, root):
        self.root = root
        self._setup_window_geometry()
        self.root.title("Добро пожаловать в Filmore!")
        self.root.configure(bg='pink2')

        self.bg_color = 'pink2'
        self.frame_color = 'PaleVioletRed3'
        self.button_color = 'VioletRed3'
        self.text_color = 'old lace'
        self.font = ('Arial', self.scale_font(15))
        self.title_font = ('Arial', self.scale_font(20), 'bold')

        self._initialize_data()
        self._setup_ui()

    def _setup_window_geometry(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.scale_factor = min(1.0, screen_width / 1920)
        window_height = int(screen_height * 0.9)
        window_width = int(screen_width * 0.9)
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.minsize(int(800 * self.scale_factor), int(600 * self.scale_factor))

    def scale_font(self, size):
        return int(size * self.scale_factor)

    def scale_value(self, value):
        return int(value * self.scale_factor)

    def _initialize_data(self):
        from data_processing import MovieLensDataPreprocessor
        from algorithms import CollaborativeFiltering

        self.data_preprocessor = MovieLensDataPreprocessor(
            data_path='ml-latest/',
            sample_fraction=0.1,
            dev_mode=False
        )
        self.prepared_data = self.data_preprocessor.run_pipeline()
        self.cf_model = CollaborativeFiltering(self.prepared_data, k=20)
        self.cf_model.fit(model_type='both')

        self.user_profile = None
        self.current_user = None
        self.users_file = "users.json"
        self.users = {}
        self.load_users()


    def _on_mousewheel(self, event):
        """Обработчик прокрутки колесиком мыши"""
        if self.canvas.winfo_exists():  # Проверяем, что canvas еще существует
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _setup_ui(self):
        self.main_container = tk.Frame(self.root, bg=self.bg_color)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            self.main_container,
            bg=self.bg_color,
            highlightthickness=0
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(
            self.main_container,
            command=self.canvas.yview,
            orient=tk.VERTICAL
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")
        ))

        self.content_frame = tk.Frame(self.canvas, bg=self.bg_color)
        self.canvas.create_window((0, 0), window=self.content_frame, anchor="nw")
        self.content_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Заменяем bind_all на bind только для canvas и content_frame
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.content_frame.bind("<MouseWheel>", self._on_mousewheel)

        self.create_main_menu()

    def create_main_menu(self):
        self.clear_window()
        buttonframe1 = tk.Frame(self.root, bg=self.frame_color)
        buttonframe1.pack(padx=10, pady=40)

        tk.Label(buttonframe1, text="Добро пожаловать в Filmore!",
                 font=self.title_font, fg=self.text_color, bg=self.button_color
                 ).pack(padx=10, pady=10)

        buttonframe2 = tk.Frame(self.root, bg=self.bg_color)
        buttonframe2.pack(padx=10, pady=20)
        tk.Label(buttonframe2, text="", font=self.title_font, bg=self.bg_color).pack(padx=10, pady=10)

        buttonframe = tk.Frame(self.root, bg=self.frame_color)
        buttonframe.pack()

        tk.Button(buttonframe, text='Войти', font=self.font, height=2, width=14,
                  fg=self.text_color, bg=self.button_color, command=self.show_login
                  ).grid(row=0, column=0, padx=10, pady=10)

        tk.Button(buttonframe, text='Регистрация', font=self.font, height=2, width=14,
                  fg=self.text_color, bg=self.button_color, command=self.show_register
                  ).grid(row=1, column=0, padx=10, pady=10)

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        if hasattr(self, 'canvas'):
            del self.canvas

    def show_login(self):
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

    def fix_article(self, title):
        if ', The' in title:
            return 'The ' + title.replace(', The', '')
        if ', A' in title:
            return 'A ' + title.replace(', A', '')
        if ', An' in title:
            return 'An ' + title.replace(', An', '')
        return title

    def show_top_movies(self):
        self.clear_window()

        main_container = tk.Frame(self.root, bg=self.bg_color)
        main_container.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)

        tk.Label(main_container, text="Топ фильмов", font=self.title_font,
                 fg=self.text_color, bg=self.button_color).pack(pady=10)

        control_frame = tk.Frame(main_container, bg=self.bg_color)
        control_frame.pack(pady=5)

        tk.Label(control_frame, text="Критерий:", font=self.font, bg=self.bg_color).grid(row=0, column=0, padx=5,
                                                                                         pady=5,
                                                                                         sticky='e')
        criteria = ['По популярности', 'По жанру', 'По году']
        self.criterion_var = tk.StringVar(value=criteria[0])
        criterion_menu = tk.OptionMenu(control_frame, self.criterion_var, *criteria)
        criterion_menu.config(font=self.font, bg=self.button_color, fg=self.text_color)
        criterion_menu.grid(row=0, column=1, padx=5, sticky='w')

        tk.Label(control_frame, text="Кол-во:", font=self.font, bg=self.bg_color).grid(row=1, column=0, padx=5, pady=5,
                                                                                       sticky='e')
        self.top_n_entry = tk.Entry(control_frame, font=self.font, width=5)
        self.top_n_entry.insert(0, "5")
        self.top_n_entry.grid(row=1, column=1, padx=5, sticky='w')

        self.params_frame = tk.Frame(control_frame, bg=self.bg_color)
        self.params_frame.grid(row=2, columnspan=2, pady=5)

        self.genre_frame = tk.Frame(self.params_frame, bg=self.bg_color)
        tk.Label(self.genre_frame, text="Жанр:", font=self.font, bg=self.bg_color).pack(side=tk.LEFT)
        all_genres = sorted(set(np.concatenate(self.data_preprocessor.movies['genres_list'].values)))
        self.genre_var = tk.StringVar(value=all_genres[0] if all_genres else '')
        genre_menu = tk.OptionMenu(self.genre_frame, self.genre_var, *all_genres)
        genre_menu.config(font=self.font, bg=self.button_color, fg=self.text_color)
        genre_menu.pack(side=tk.LEFT, padx=5)

        self.year_frame = tk.Frame(self.params_frame, bg=self.bg_color)
        tk.Label(self.year_frame, text="Год:", font=self.font, bg=self.bg_color).pack(side=tk.LEFT)
        self.year_entry = tk.Entry(self.year_frame, font=self.font, width=6)
        self.year_entry.pack(side=tk.LEFT, padx=5)

        btn_frame = tk.Frame(control_frame, bg=self.bg_color)
        btn_frame.grid(row=3, columnspan=2, pady=10)
        tk.Button(btn_frame, text="Найти", font=self.font, fg=self.text_color,
                  bg=self.button_color, command=self._perform_top_search).pack()

        result_frame = tk.Frame(main_container, bg=self.bg_color)
        result_frame.pack(pady=(10, 5), fill=tk.BOTH, expand=True)

        self.result_text = tk.Text(result_frame, font=self.font, height=8, width=50, wrap=tk.WORD)
        scrollbar = tk.Scrollbar(result_frame, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.result_text.config(state='disabled')

        tk.Button(main_container, text="Назад", font=self.font, fg=self.text_color,
                  bg=self.button_color, command=self.show_recommendation_interface).pack(pady=10)

        self._update_top_movies_fields()
        self.criterion_var.trace('w', lambda *args: self._update_top_movies_fields())

    def _update_top_movies_fields(self):
        for widget in self.params_frame.winfo_children():
            widget.pack_forget()

        if self.criterion_var.get() == 'По жанру':
            self.genre_frame.pack(pady=5)
        elif self.criterion_var.get() == 'По году':
            self.year_frame.pack(pady=5)

    def _perform_top_search(self):
        try:
            top_n = int(self.top_n_entry.get())
            if not 1 <= top_n <= 20:
                raise ValueError("Количество фильмов должно быть от 1 до 20")
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректное число от 1 до 20", parent=self.root)
            return

        try:
            criterion = self.criterion_var.get()
            results = None

            if criterion == 'По популярности':
                results = self.data_preprocessor.get_top_by_popularity(top_n=top_n, min_ratings=50)
            elif criterion == 'По жанру':
                genre = self.genre_var.get()
                results = self.data_preprocessor.get_top_by_genre(genre=genre, top_n=top_n, min_ratings=50)
            elif criterion == 'По году':
                year = int(self.year_entry.get())
                results = self.data_preprocessor.get_top_by_year(year=year, top_n=top_n, min_ratings=50)

            self.result_text.config(state='normal')
            self.result_text.delete(1.0, tk.END)

            if results.empty:
                self.result_text.insert(tk.END, "Фильмы не найдены\n")
            else:
                for _, row in results.iterrows():
                    fixed_title = self.fix_article(row['title'])
                    self.result_text.insert(tk.END, f"{fixed_title}\n")
                    if 'year' in row and not pd.isna(row['year']):
                        self.result_text.insert(tk.END, f"Год: {int(row['year'])}\n")
                    self.result_text.insert(tk.END, f"Жанры: {row['genres']}\n")
                    if 'rating_metric' in row:
                        metric_name = "Рейтинг" if criterion == 'По жанру' else "Количество оценок"
                        self.result_text.insert(tk.END, f"{metric_name}: {row['rating_metric']:.2f}\n")
                    self.result_text.insert(tk.END, "\n")

        except Exception as e:
            messagebox.showerror("Ошибка", str(e), parent=self.root)
        finally:
            self.result_text.config(state='disabled')

    def logout(self):
        self.save_profile()
        self.current_user = None
        self.user_profile = None
        self.create_main_menu()

    def save_profile(self):
        if self.current_user and self.user_profile:
            self.users[self.current_user]['profile'] = self.user_profile
            self.save_users()

    def load_users(self):
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r', encoding='utf-8') as f:
                    self.users = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                self.users = {}

    def save_users(self):
        with open(self.users_file, 'w', encoding='utf-8') as f:
            json.dump(self.users, f, ensure_ascii=False, indent=4)

    def run_user_survey(self):
        self.clear_window()

        tk.Label(self.root, text="Шаг 1: Выберите любимые жанры", font=self.title_font,
                 fg=self.text_color, bg=self.button_color).pack(pady=10)

        genres_frame = tk.Frame(self.root, bg=self.bg_color)
        genres_frame.pack(pady=5)

        all_genres = sorted(set(np.concatenate(self.data_preprocessor.movies['genres_list'].values)))
        self.genre_vars = []
        for i, genre in enumerate(all_genres):
            var = tk.BooleanVar()
            self.genre_vars.append((genre, var))
            tk.Checkbutton(genres_frame, text=genre, variable=var, font=self.font, bg=self.bg_color).grid(
                row=i // 4, column=i % 4, padx=5, pady=5)

        tk.Button(self.root, text="Далее →", font=self.font, fg=self.text_color,
                  bg=self.button_color, command=lambda: self._load_movies_for_survey()).pack(pady=10)

    def _load_movies_for_survey(self):
        selected_genres = [genre for genre, var in self.genre_vars if var.get()]
        if not selected_genres:
            messagebox.showerror("Ошибка", "Выберите хотя бы один жанр", parent=self.root)
            return

        try:
            top_movies = self.data_preprocessor.get_top_by_genre(
                genre=selected_genres[0],
                top_n=5,
                min_ratings=50
            )
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить фильмы: {str(e)}", parent=self.root)
            return

        self.clear_window()
        tk.Label(self.root, text="Шаг 2: Оцените фильмы (1-5)", font=self.title_font,
                 fg=self.text_color, bg=self.button_color).pack(pady=10)

        movies_frame = tk.Frame(self.root, bg=self.bg_color)
        movies_frame.pack(pady=5)

        self.movie_ratings = {}
        for _, row in top_movies.iterrows():
            title = self.fix_article(row['title'])
            tk.Label(movies_frame, text=title, font=self.font, bg=self.bg_color).pack(anchor='w')
            rating_entry = tk.Entry(movies_frame, font=self.font, width=5)
            rating_entry.pack(anchor='w')
            self.movie_ratings[row['movieId']] = rating_entry

        tk.Label(self.root, text="Диапазон годов (например, 1990-2020):", font=self.font, bg=self.bg_color).pack(pady=5)
        years_frame = tk.Frame(self.root, bg=self.bg_color)
        years_frame.pack(pady=5)

        tk.Label(years_frame, text="От:", font=self.font, bg=self.bg_color).pack(side=tk.LEFT)
        self.year_from_entry = tk.Entry(years_frame, font=self.font, width=6)
        self.year_from_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(years_frame, text="До:", font=self.font, bg=self.bg_color).pack(side=tk.LEFT)
        self.year_to_entry = tk.Entry(years_frame, font=self.font, width=6)
        self.year_to_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(self.root, text="Минимальный рейтинг (0.5-5.0):", font=self.font, bg=self.bg_color).pack(pady=5)
        self.min_rating_entry = tk.Entry(self.root, font=self.font, width=10)
        self.min_rating_entry.insert(0, "3.0")
        self.min_rating_entry.pack(pady=5)

        tk.Button(self.root, text="Сохранить", font=self.font, fg=self.text_color,
                  bg=self.button_color, command=self.save_survey).pack(pady=10)

        tk.Button(self.root, text="Назад", font=self.font, fg=self.text_color,
                  bg=self.button_color, command=self.show_recommendation_interface).pack(pady=10)

    def save_survey(self):
        try:
            self.user_profile['favorite_genres'] = [genre for genre, var in self.genre_vars if var.get()]

            year_from = int(self.year_from_entry.get()) if self.year_from_entry.get() else 1900
            year_to = int(self.year_to_entry.get()) if self.year_to_entry.get() else datetime.now().year
            self.user_profile['favorite_years'] = list(range(year_from, year_to + 1))

            min_rating = float(self.min_rating_entry.get())
            if not 0.5 <= min_rating <= 5.0:
                raise ValueError("Рейтинг должен быть от 0.5 до 5.0")
            self.user_profile['min_rating'] = min_rating

            self.user_profile['watched_movies'] = {}
            for movie_id, entry in self.movie_ratings.items():
                rating = entry.get()
                if rating:
                    rating = float(rating)
                    if 1.0 <= rating <= 5.0:
                        self.user_profile['watched_movies'][str(movie_id)] = rating

            new_ratings = pd.DataFrame([
                {
                    'userId': self.current_user,
                    'movieId': int(movie_id),
                    'rating': float(rating),
                    'timestamp': int(datetime.now().timestamp())
                }
                for movie_id, rating in self.user_profile['watched_movies'].items()
            ])

            self.cf_model.data['ratings'] = pd.concat([
                self.cf_model.data['ratings'],
                new_ratings
            ]).drop_duplicates(['userId', 'movieId'])

            print("Обновление маппингов...")
            unique_users = self.cf_model.data['ratings']['userId'].unique()
            unique_movies = self.cf_model.data['ratings']['movieId'].unique()

            self.cf_model.data['user_mapping'] = {user: idx for idx, user in enumerate(unique_users)}
            self.cf_model.data['movie_mapping'] = {movie: idx for idx, movie in enumerate(unique_movies)}

            print(f"Обновлено пользователей: {len(self.cf_model.data['user_mapping'])}")
            print(f"Обновлено фильмов: {len(self.cf_model.data['movie_mapping'])}")

            self.cf_model.fit(model_type='both')

            self.save_profile()
            messagebox.showinfo("Успех", "Профиль сохранен!", parent=self.root)
            self.show_recommendation_interface()

        except ValueError as e:
            messagebox.showerror("Ошибка", f"Некорректные данные: {e}", parent=self.root)
        except Exception as e:
            messagebox.showerror("Критическая ошибка", f"Ошибка при обновлении маппингов: {str(e)}", parent=self.root)

    def show_recommendations(self):
        if not self.user_profile:
            messagebox.showwarning("Предупреждение", "Профиль пользователя не загружен. Сначала пройдите опрос.",
                                   parent=self.root)
            return

        if not self.user_profile.get('watched_movies'):
            messagebox.showwarning("Предупреждение", "Вы не оценили ни одного фильма. Сначала пройдите опрос.",
                                   parent=self.root)
            return

        self.clear_window()

        tk.Label(self.root, text="Рекомендации для вас", font=self.title_font,
                 fg=self.text_color, bg=self.button_color).pack(pady=10)

        try:
            survey_answers = {int(k): float(v) for k, v in self.user_profile['watched_movies'].items()}

            recommendations = self.cf_model.recommend_for_new_user(
                survey_answers,
                n=10,
                model_type='hybrid',
                user_profile=self.user_profile
            )

            if recommendations.empty:
                raise ValueError("Нет рекомендаций")

            movie_stats = (
                self.data_preprocessor.ratings.groupby('movieId')
                .agg(avg_rating=('rating', 'mean'), rating_count=('rating', 'count'))
                .reset_index()
            )

            recommendations = recommendations.merge(
                movie_stats[['movieId', 'avg_rating', 'rating_count']],
                on='movieId',
                how='left'
            )

        except Exception as e:
            print(f"Ошибка: {e}")
            recommendations = self.cf_model._get_fallback_recommendations(10)
            recommendations = recommendations.merge(
                self.data_preprocessor.prepared_data['movie_data'][['movieId', 'avg_rating', 'rating_count']],
                on='movieId',
                how='left'
            )

        text_area = tk.Text(self.root, font=self.font, height=15, width=50)
        text_area.pack(pady=10)

        if recommendations.empty:
            text_area.insert(tk.END, "Не удалось найти рекомендации. Попробуйте оценить больше фильмов.\n")
        else:
            for _, row in recommendations.iterrows():
                fixed_title = self.fix_article(row['title'])
                text_area.insert(tk.END, f"{fixed_title} ({row['genres']})\n")
                if not pd.isna(row['avg_rating']):
                    text_area.insert(tk.END,
                                     f"Средний рейтинг: {row['avg_rating']:.2f} (на основе {int(row['rating_count'])} оценок)\n\n")
                else:
                    text_area.insert(tk.END, "Нет данных о рейтинге\n\n")

        text_area.config(state='disabled')

        tk.Button(self.root, text="Назад", font=self.font, fg=self.text_color,
                  bg=self.button_color, command=self.show_recommendation_interface).pack(pady=10)


if __name__ == "__main__":
    root = tk.Tk()
    app = MovieRecommendationApp(root)
    root.mainloop()
