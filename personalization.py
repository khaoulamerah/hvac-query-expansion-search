from collections import Counter

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

DECAY = 0.9  # a search/click is worth 90% of the next more recent one
INTEREST_WEIGHT = 2.0  # an "Interested" click counts more than a plain search


def build_profile(idx, tokenize, queries, product_names) -> np.ndarray:
    """queries and product_names must be ordered newest first. Returns a vector over idx.vocab."""
    profile = np.zeros(len(idx.vocab))
    for rank, q in enumerate(queries):
        for term in tokenize(q):
            j = idx.vocab.get(term)
            if j is not None:
                profile[j] += DECAY**rank
    for rank, name in enumerate(product_names):
        i = idx.pos.get(name)
        if i is not None:
            profile += INTEREST_WEIGHT * DECAY**rank * idx.X[i]
    return profile


def affinity(idx, profile: np.ndarray) -> np.ndarray:
    """Cosine similarity between the profile and every product (zeros if the profile is empty)."""
    if not profile.any():
        return np.zeros(idx.N)
    return cosine_similarity(profile.reshape(1, -1), idx.X)[0]


def top_terms(idx, profile: np.ndarray, n: int = 6) -> list[str]:
    """Keywords (stemmed) the user is most interested in."""
    if not profile.any():
        return []
    return [idx.terms[j] for j in np.argsort(-profile)[:n] if profile[j] > 0]


def top_categories(idx, product_names, n: int = 3) -> list[str]:
    """Favourite categories, based on the products marked 'Interested'."""
    cats = Counter(
        idx.df.at[idx.pos[p], "category"] for p in product_names if p in idx.pos
    )
    return [c for c, _ in cats.most_common(n)]
