import re
from collections import Counter
from typing import Tuple, List, Optional, Set

import pandas as pd
from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer, FunctionTransformer, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


DEFAULT_THRESHOLD = 200
DEFAULT_TOP_N = 30
DEFAULT_SKIP = {
    "ปากซอย", "ซอย", "บริเวณ", "หน้า", "บ้าน", "คน", "เขต", "จุด",
    'ขอบคุณ', 'เมตร', 'เข้ามา', 'แยก', 'เลขที่', 'เดิน',
    "ทางเท้า", "ฟุตบาท", "ทางเดิน", "เจ้าหน้าที่", "ประชาชน", "#1555", "เวลา"
}


class SimpleMLB(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def _parse(self, s):
        if s is None:
            return []
        if isinstance(s, (list, set, tuple)):
            return list(s)
        s = str(s).strip()
        if s in ("", "{}"):
            return []
        if s.startswith("{") and s.endswith("}"):
            s = s[1:-1]
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return parts

    def fit(self, X, y=None):
        lists = [self._parse(x) for x in X]
        self.mlb.fit(lists)
        return self

    def transform(self, X):
        lists = [self._parse(x) for x in X]
        return self.mlb.transform(lists)

    def get_feature_names_out(self, input_features=None):
        return [f"type__{c}" for c in self.mlb.classes_]


def extract_datetime(df_slice: pd.DataFrame) -> pd.DataFrame:

    ts = pd.to_datetime(df_slice.iloc[:, 0], errors='coerce')
    return pd.DataFrame({
        'year': ts.dt.year.fillna(-1).astype(int),
        'month': ts.dt.month.fillna(-1).astype(int),
        'day': ts.dt.day.fillna(-1).astype(int),
        'weekday': ts.dt.weekday.fillna(-1).astype(int),
        'hour': ts.dt.hour.fillna(-1).astype(int),
        'is_weekend': (ts.dt.weekday >= 5).fillna(False).astype(int)
    })


def _clean_token(w: str) -> bool:
    if not w or str(w).strip() == "":
        return False
    if re.fullmatch(r"[\W_]+", str(w)):
        return False
    if re.fullmatch(r"\d+", str(w)):
        return False
    return True


def _compute_keyword_set_from_df(
    df: pd.DataFrame,
    comment_col: str = 'comment',
    top_n: int = DEFAULT_TOP_N,
    skip: Optional[Set[str]] = None,
    stopwords: Optional[Set[str]] = None,
) -> List[str]:

    if skip is None:
        skip = DEFAULT_SKIP
    if stopwords is None:
        stopwords = set(thai_stopwords())

    df[comment_col] = df.get(comment_col, pd.Series(index=df.index)).fillna("")

    df['tokens'] = df[comment_col].fillna('').apply(word_tokenize)
    df['tokens_clean'] = df['tokens'].apply(lambda toks: [w for w in toks if _clean_token(w)])
    df['tokens_no_stop'] = df['tokens_clean'].apply(lambda toks: [w for w in toks if w not in stopwords])

    all_tokens = [w for toks in df['tokens_no_stop'] for w in toks]
    tok_counter = Counter(all_tokens)
    candidates = [(w, c) for w, c in tok_counter.most_common() if w not in skip]
    important_words = [w for w, _ in candidates[:top_n]]

    important_set = set(important_words)
    df['comment_keywords'] = df['tokens_no_stop'].apply(lambda toks: ' '.join([w for w in toks if w in important_set]))
    df['comment_keywords_list'] = df['tokens_no_stop'].apply(lambda toks: [w for w in toks if w in important_set])

    return important_words


def _apply_subdistrict_threshold(df: pd.DataFrame, threshold: int = DEFAULT_THRESHOLD, col: str = 'subdistrict') -> None:

    df[col] = df[col].fillna('Missing')
    counts = df[col].value_counts()
    df[col] = df[col].apply(lambda x: x if counts.get(x, 0) >= threshold else 'Other')


def build_pipeline(max_tfidf_features: int = 200) -> Pipeline:

    tfidf_for_keywords = TfidfVectorizer(
        tokenizer=lambda x: x.split(),   # comment_keywords is already tokenized into words separated by spaces
        preprocessor=lambda x: x,
        token_pattern=None,
        lowercase=False,
        max_features=max_tfidf_features
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('datetime', FunctionTransformer(extract_datetime, validate=False), ['timestamp']),
            ('type', SimpleMLB(), 'type'),
            ('comment', tfidf_for_keywords, 'comment_keywords'),
            ('coords', 'passthrough', ['lon', 'lat']),
            ('subdistrict', OneHotEncoder(handle_unknown='ignore'), ['subdistrict']),
        ],
        remainder='drop',
        sparse_threshold=0.0
    )

    pipeline = Pipeline([('preprocess', preprocessor)])
    return pipeline


def pipeline_to_df(pipeline: Pipeline, df: pd.DataFrame) -> pd.DataFrame:

    try:
        X = pipeline.transform(df)
    except Exception:
        X = pipeline.fit_transform(df)

    ct = pipeline.named_steps['preprocess']

    # datetime names
    dt_cols = ['year', 'month', 'day', 'weekday', 'hour', 'is_weekend']

    # type names
    type_cols = ct.named_transformers_['type'].get_feature_names_out()

    # tfidf names
    tfidf_vect = ct.named_transformers_['comment']
    tfidf_cols = [f"tfidf__{c}" for c in tfidf_vect.get_feature_names_out().tolist()]

    # coord cols
    coord_cols = ['lon', 'lat']

    sub_enc = ct.named_transformers_['subdistrict']
    sub_cols = [f"sub__{c}" for c in sub_enc.get_feature_names_out()]

    colnames = list(dt_cols) + list(type_cols) + tfidf_cols + coord_cols + sub_cols

    if hasattr(X, "toarray"):
        X = X.toarray()

    return pd.DataFrame(X, columns=colnames, index=df.index)


def preprocess_df_for_ml(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    comment_col: str = 'comment',
    threshold: int = DEFAULT_THRESHOLD,
    top_n: int = DEFAULT_TOP_N,
    skip: Optional[Set[str]] = None,
    stopwords: Optional[Set[str]] = None,
    max_tfidf_features: int = 200,
) -> Tuple[pd.DataFrame, Pipeline]:

    df_copy = df.copy()

    if timestamp_col not in df_copy.columns:
        raise KeyError(f"timestamp column '{timestamp_col}' not found in DataFrame")
    if comment_col not in df_copy.columns:
        df_copy[comment_col] = ''

    _apply_subdistrict_threshold(df_copy, threshold=threshold, col='subdistrict')

    important_words = _compute_keyword_set_from_df(
        df_copy,
        comment_col=comment_col,
        top_n=top_n,
        skip=skip,
        stopwords=stopwords,
    )

    pipeline = build_pipeline(max_tfidf_features=max_tfidf_features)
    pipeline.fit(df_copy)
    preprocessed_df = pipeline_to_df(pipeline, df_copy)

    return preprocessed_df, pipeline

__all__ = [
    'SimpleMLB',
    'extract_datetime',
    'build_pipeline',
    'pipeline_to_df',
    'preprocess_df_for_ml',
]
