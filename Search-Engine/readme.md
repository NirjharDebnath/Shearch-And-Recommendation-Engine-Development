# Hybrid Anime Search Algorithm 🎯

### This Code Is A Part Of The SEARCH AND RECOMMENDATION ENGINE DEVELOPMENT (SARED) Project
**A hybrid search engine for anime that allows users to search by title or genre, combining multiple techniques like BERT embeddings, TF-IDF, fuzzy matching, and FAISS indexing for high-performance results.**
<hr>


## 🚀 Features
**Search by Title**: Uses BERT embeddings, TF-IDF, and Fuzzy Matching for accurate anime title search.

**Search by Genre**: Finds anime with the most common genres while handling typos and variations.

**FAISS Indexing**: Enables fast similarity search for title-based queries.

**Popularity & Rating Boosting**: Ranks results by adjusting scores based on popularity and average rating.

**Typos Handling**: Uses fuzzy string matching to correct typos in genres.

## 📦 Installation
1️⃣ Install Dependencies
```bash
pip install pandas numpy faiss-cpu scikit-learn fuzzywuzzy sentence-transformers
```
2️⃣ Download Pretrained BERT Model
The model used for encoding anime titles is MiniLM-L6-v2, available via sentence-transformers:

```python
from sentence_transformers import SentenceTransformer
bert = SentenceTransformer("paraphrase-MiniLM-L6-v2")
```
## 📝 How It Works
### The search engine has two modes:

### 🔍 Title-Based Search<br>

- **FAISS (BERT)**: Finds the most similar anime titles using BERT embeddings.

- **TF-IDF Similarity**: Measures how well the query matches the anime descriptions.

- **Fuzzy Matching**: Handles typos & similar spellings in titles.

- **Hybrid Score**: Combines all scores, adding popularity and rating boosts.

### 🎭 Genre-Based Search
- **Genre Cleaning**: Converts genres to lowercase and removes extra spaces.

- **Fuzzy Matching for Typos**: Fixes misspellings in user-provided genres.

- **Genre Overlap Score**: Measures how many genres match between the query and anime.

- **Hybrid Score**: Adjusts scores using popularity & rating normalization.

## 📌 Usage
### 🎯 Initialize the Search Engine
```python
hybrid_search = Hybrid_Search()
```
### 🔍 Search by Title
```python
result = hybrid_search.search(query="Attack on Titan", search_by="title")
print(result)
```
### 🎭 Search by Genre
```python
result = hybrid_search.search(query="acti0n, adventrue", search_by="genre")  # Typo in genres
print(result)
```
- **✅ Corrects typos** ("acti0n" → "Action", "adventrue" → "Adventure")
- **✅ Finds the most relevant anime** based on genre similarity, popularity, and ratings

## 💡 Code Breakdown
**🔹 Class: Hybrid_Search**
- Loads the anime dataset.

- Prepares TF-IDF and BERT embeddings.

- Creates FAISS Index for fast similarity searches.

- Stores a set of unique genres for fuzzy matching.

**🔹 search_by_title(query)**
- Finds BERT similarity using FAISS.

- Computes TF-IDF similarity.

- Uses Fuzzy Matching to improve results.

- Combines scores and normalizes based on popularity & rating.

**🔹 search_by_genre(query)**
- Cleans and standardizes genres.

- Fixes genre typos using Fuzzy Matching.

- Finds anime with maximum genre overlap.

- Adjusts scores based on popularity and rating.

### 📊 Score Calculation
**The final Hybrid Score is calculated as**:

```python
Hybrid Score = (BERT Score * 0.5) + (TF-IDF Score * 0.3) + (Fuzzy Score * 0.2) 
             + (Popularity Score * 0.1) + (Rating Score * 0.2)
```
Where:

- **BERT Score**: Measures semantic similarity of titles.

- **TF-IDF Score**: Measures text similarity of descriptions.

- **Fuzzy Score**: Helps with typos in the title.

- **Popularity & Rating**: Ensures high-quality recommendations.

**For genre-based search, the Hybrid Score is computed as**:

```python
Hybrid Score = (Genre Overlap Score) + (Popularity Score * 0.1) + (Rating Score * 0.2)
```
### ⚡ Performance Optimization
- **FAISS for Fast Similarity Search**: Accelerates title-based search.

- **Efficient Genre Matching**: Uses sets for quick lookup.

- **Fuzzy Matching for Robust Search**: Corrects typos in both titles and genres.

### 📚 Future Improvements
- Support for Multi-Keyword Queries (e.g., "Naruto + Action + 2005").

- More Ranking Factors (e.g., number of episodes, studio).

- Better Handling of Synonyms (e.g., "Sci-Fi" → "Science Fiction").

## 💻 Author
Developed by **Nirjhar**, focusing on AI-powered anime search. 🚀
**For improvements and contributions, feel free to fork this repo and submit a pull request!✨**
<hr>
