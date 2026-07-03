from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def chunk_text(text, chunk_size=120, overlap=20):
    """Split text into overlapping word chunks for retrieval."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_index(chunks):
    """Build a TF-IDF index over document chunks."""
    if not chunks:
        return None, None
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(chunks)
    return vectorizer, matrix


def retrieve(query, vectorizer, matrix, chunks, top_k=3):
    """Return top-k most relevant chunks for a query."""
    if vectorizer is None or matrix is None:
        return []
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).flatten()
    ranked = scores.argsort()[::-1][:top_k]
    return [chunks[i] for i in ranked if scores[i] > 0]