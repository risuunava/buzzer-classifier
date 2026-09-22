"""
Fitur linguistik: kemiripan teks antar komentar, skor sentimen,
rasio emoji/hashtag, panjang komentar.
"""
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

stemmer = StemmerFactory().create_stemmer()
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U0001F600-\U0001F64F"
    "]+",
    flags=re.UNICODE,
)


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text


def add_linguistic_features(comments_df: pd.DataFrame) -> pd.DataFrame:
    df = comments_df.copy()

    df["comment_length"] = df["comment_text"].str.len()
    df["emoji_count"] = df["comment_text"].apply(lambda t: len(EMOJI_PATTERN.findall(str(t))))
    df["hashtag_count"] = df["comment_text"].apply(lambda t: str(t).count("#"))
    df["emoji_hashtag_ratio"] = (df["emoji_count"] + df["hashtag_count"]) / df["comment_length"].replace(0, 1)

    df["clean_text"] = df["comment_text"].apply(preprocess_text)

    # Kemiripan antar komentar (rata-rata cosine similarity terhadap komentar lain)
    if len(df) > 1:
        tfidf = TfidfVectorizer().fit_transform(df["clean_text"])
        sim_matrix = cosine_similarity(tfidf)
        # rata-rata similarity ke komentar lain (exclude diri sendiri)
        df["avg_similarity_to_others"] = (sim_matrix.sum(axis=1) - 1) / (len(df) - 1)
    else:
        df["avg_similarity_to_others"] = 0.0

    # TODO: skor sentimen -- bisa pakai IndoBERT sentiment model dari HuggingFace
    df["sentiment_score"] = None

    return df
