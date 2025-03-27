import fuzzywuzzy.process as process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer   

Data_Path = "/home/nirjhar/Python Codes/Machine Learning/ARS/cleaned_anilist_anime_2.csv"

class Hybrid_Search():
    def __init__(self):
        super(Hybrid_Search, self).__init__()
        
        self.data = pd.read_csv(Data_Path)

        self.data["Genres"] = self.data["Genres"].apply(lambda x: self._clean_genres(x))
        self.unique_genres = {genre for genres in self.data["Genres"] for genre in genres}


        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.data["Description"])

        self.bert = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        self.titles = self.data["Title"].astype(str).tolist()
        self.title_embeddings = self.bert.encode(self.titles, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(self.title_embeddings.shape[1])
        self.index.add(self.title_embeddings)

    def _clean_genres(self, genre_str):

        if not isinstance(genre_str, str):
            return set()
        return set(map(str.strip, genre_str.lower().split(", "))) 

    def _fix_genre_typos(self, query_genres):

        fixed_genres = set()
        for genre in query_genres:
            match, score = process.extractOne(genre, self.unique_genres) if self.unique_genres else (genre, 100)
            fixed_genres.add(match if score > 80 else genre) 

    def search(self, query, search_by="title", top_n=5, popularity_weight=0.1, rating_weight=0.2):
        if search_by == "title":
            return self._search_by_title(query, top_n, popularity_weight, rating_weight)
        elif search_by == "genre":
            return self._search_by_genre(query, top_n, popularity_weight, rating_weight)
        else:
            raise ValueError("Invalid search_by parameter. Use 'title' or 'genre'.")

    def _search_by_title(self, query, top_n, popularity_weight, rating_weight):
        query_embedding = self.bert.encode([query], convert_to_numpy=True)
        _, top_indices = self.index.search(query_embedding, top_n)
        bert_sim = np.zeros(len(self.data))
        bert_sim[top_indices[0]] = 1
        
        query_tfidf = self.tfidf_vectorizer.transform([query])
        tfidf_sim = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()

        fuzzy_matches = process.extract(query, self.data["Title"], limit=len(self.data))
        max_fuzzy_score = max(score for _, score, _ in fuzzy_matches)
        fuzzy_scores = {title: score / max_fuzzy_score for title, score, _ in fuzzy_matches}
        
        self.data["Hybrid Score"] = [
            (0.5 * bert_sim[i]) + (0.3 * tfidf_sim[i]) + (0.2 * fuzzy_scores.get(title, 0)) 
            for i, title in enumerate(self.data["Title"])
        ]

        self._apply_normalization(popularity_weight, rating_weight)
        return self.data.nlargest(top_n, "Hybrid Score")

    def _search_by_genre(self, query, top_n, popularity_weight, rating_weight):
        query_genres = self._clean_genres(query) 
        query_genres = self._fix_genre_typos(query_genres) 

        # Compute genre overlap
        self.data["Genre Score"] = self.data["Genres"].apply(lambda genres: len(query_genres & genres) / max(len(query_genres), 1))

        self.data["Hybrid Score"] = self.data["Genre Score"]
        self._apply_normalization(popularity_weight, rating_weight)

        return self.data.nlargest(top_n, "Hybrid Score")

    def _apply_normalization(self, popularity_weight, rating_weight):
        max_popularity = self.data["popularity"].max()
        max_rating = self.data["averageScore"].max()

        if max_popularity > 0:
            self.data["Hybrid Score"] += (popularity_weight * self.data["popularity"] / max_popularity)
        if max_rating > 0:
            self.data["Hybrid Score"] += (rating_weight * self.data["averageScore"] / max_rating)

# object generation
hybrid_search = Hybrid_Search()
