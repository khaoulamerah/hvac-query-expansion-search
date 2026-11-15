import math
import re
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import db
import personalization

# 1. DATASET 
CATALOG_PATH = "data/products.csv"
REQUIRED_COLUMNS = {"name", "category", "description", "price"}


def load_products(uploaded=None) -> pd.DataFrame:
    df = pd.read_csv(uploaded if uploaded is not None else CATALOG_PATH)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"The CSV must contain the columns: {', '.join(sorted(REQUIRED_COLUMNS))}. Missing: {', '.join(sorted(missing))}"
        )
    for col in ("name", "category", "description"):
        df[col] = df[col].fillna("").astype(str)
    return df


# KNOWLEDGE BASES for expansion (domain dictionaries)
SYNONYMS = {
    "boiler": ["heater", "water heater"],
    "faucet": ["tap", "mixer"],
    "tap": ["faucet", "mixer"],
    "pipe": ["tube", "piping"],
    "radiator": ["heater", "towel rail"],
    "leak": ["drip", "burst"],
    "pump": ["circulator", "circulation"],
    "thermostat": ["controller"],
    "valve": ["stopcock", "shutoff"],
    "toilet": ["wc", "closet"],
    "sink": ["basin"],
    "basin": ["sink"],
    "tube": ["pipe"],
}
RELATED = {
    "boiler": ["thermostat", "flue", "filter", "expansion vessel"],
    "radiator": ["valve", "bleed", "thermostat"],
    "faucet": ["cartridge", "sealant"],
    "tap": ["cartridge", "sealant"],
    "pipe": ["fitting", "elbow", "coupling", "insulation"],
    "leak": ["sealant", "tape", "clamp", "cartridge"],
    "shower": ["mixer", "thermostatic"],
    "toilet": ["cistern", "seat"],
}
CONCEPTS = {  # high-level concept -> concrete product terms
    "heating": ["boiler", "radiator", "thermostat", "underfloor", "pump"],
    "plumbing": ["pipe", "valve", "tap", "fitting"],
    "bathroom": ["shower", "toilet", "basin", "towel"],
    "hot water": ["boiler", "water heater", "cylinder"],
    "repair": ["clamp", "tape", "sealant", "cartridge"],
}
STOP = set(
    [
        "a",
        "an",
        "the",
        "for",
        "to",
        "of",
        "and",
        "or",
        "in",
        "on",
        "with",
        "my",
        "me",
        "i",
        "need",
        "want",
        "buy",
        "looking",
        "some",
        "any",
        "is",
        "are",
    ]
)


# IDENTIFY QUERY TERMS (tokenize, stop-words, light stemming)
def stem(w: str) -> str:
    for suf in ("ing", "ed", "s"):
        if w.endswith(suf) and not w.endswith("ss") and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


#  INDEX: BM25 + TF-IDF over the product catalogue


class Index:
    def __init__(self, df: pd.DataFrame, k1=1.5, b=0.75):
        self.df = df.reset_index(drop=True)
        self.texts = (
            self.df["name"]
            + " "
            + self.df["name"]
            + " "
            + self.df["category"]
            + " "
            + self.df["description"]
        ).tolist()
        self.docs = [tokenize(t) for t in self.texts]
        self.k1, self.b = k1, b
        self.tfs = [Counter(d) for d in self.docs]
        self.dl = np.array([len(d) for d in self.docs])
        self.avgdl = self.dl.mean()
        self.dfreq = Counter(t for d in self.docs for t in set(d))
        self.N = len(self.docs)
        self.vec = TfidfVectorizer(
            tokenizer=tokenize, lowercase=False, token_pattern=None
        )
        self.X = self.vec.fit_transform(self.texts).toarray()
        self.vocab = self.vec.vocabulary_
        self.terms = self.vec.get_feature_names_out()
        self.pos = {n: i for i, n in enumerate(self.df["name"])}  # product name -> row

    def bm25(self, weights: dict) -> np.ndarray:
        scores = np.zeros(self.N)
        for t, w in weights.items():
            if t not in self.dfreq:
                continue
            idf = math.log((self.N - self.dfreq[t] + 0.5) / (self.dfreq[t] + 0.5) + 1)
            for i, tf in enumerate(self.tfs):
                f = tf.get(t, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                    scores[i] += w * idf * f * (self.k1 + 1) / norm
        return scores

    def tfidf(self, weights: dict) -> np.ndarray:
        q = np.zeros(len(self.vocab))
        for t, w in weights.items():
            if t in self.vocab:
                q[self.vocab[t]] += w
        if not q.any():
            return np.zeros(self.N)
        return cosine_similarity(q.reshape(1, -1), self.X)[0]

    # --- helpers for automatic / feedback expansion
    def similar_terms(self, q_terms, n=3, min_sim=0.25):
        Xt, out = self.X.T, []
        for t in q_terms:
            if t in self.vocab:
                sims = cosine_similarity(Xt[self.vocab[t]].reshape(1, -1), Xt)[0]
                for j in np.argsort(-sims)[1 : n + 1]:
                    if sims[j] >= min_sim:
                        out.append(self.terms[j])
        return out

    def feedback_terms(self, doc_ids, n=5):
        centroid = self.X[doc_ids].mean(axis=0)
        return [self.terms[j] for j in np.argsort(-centroid)[:n]]


#  EXPAND QUERY TERMS
def expand_query(query: str, methods: list[str], idx: Index):
    q = tokenize(query)
    qset = set(q)
    weights = {t: 1.0 for t in q}  # original terms always weight 1.0
    log = []

    def add(terms, w, method):
        for phrase in terms:
            for t in tokenize(phrase):
                if t not in weights:
                    weights[t] = w
                    log.append({"term": t, "method": method, "weight": w})

    def lookup(kb, w, method):
        for key, vals in kb.items():
            if set(tokenize(key)) <= qset:
                add(vals, w, method)

    if "Synonym" in methods:
        lookup(SYNONYMS, 0.6, "Synonym")
    if "Related term" in methods:
        lookup(RELATED, 0.4, "Related term")
    if "Concept" in methods:
        lookup(CONCEPTS, 0.4, "Concept")
    if "Automatic (corpus)" in methods:
        add(idx.similar_terms(q), 0.3, "Automatic (corpus)")
    if "Feedback (PRF)" in methods:  # pseudo-relevance feedback: use top-3 of 1st pass
        first = idx.bm25(weights)
        top = [i for i in np.argsort(-first)[:3] if first[i] > 0]
        if top:
            add(idx.feedback_terms(top), 0.3, "Feedback (PRF)")
    return weights, log


#  RE-RANKING
@st.cache_resource(show_spinner="Loading neural model...")
def load_neural():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def minmax(x):
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng else np.zeros_like(x)


def search(
    query, methods, reranker, k, idx: Index, n_candidates=12, profile=None, alpha=0.0
):
    """Expansion -> BM25 candidates -> re-ranking -> personalization.
    """
    weights, log = expand_query(query, methods, idx)
    active = profile is not None and alpha > 0 and bool(profile.any())
    if active:  # give the history more candidates to promote
        n_candidates = max(n_candidates, 20)

    first = idx.bm25(weights)  # stage 1: candidate retrieval
    cand = [i for i in np.argsort(-first)[:n_candidates] if first[i] > 0]
    if not cand:
        return [], weights, log
    cand = np.array(cand)

    if reranker == "BM25":
        final = first[cand]
    elif reranker == "TF-IDF":
        final = idx.tfidf(weights)[cand]
    elif reranker == "Hybrid (BM25 + TF-IDF)":
        final = 0.5 * minmax(first[cand]) + 0.5 * minmax(idx.tfidf(weights)[cand])
    else:  # Neural
        model = load_neural()
        q_text = query + " " + " ".join(weights.keys())
        qe = model.encode([q_text], normalize_embeddings=True)
        de = model.encode([idx.texts[i] for i in cand], normalize_embeddings=True)
        final = (qe @ de.T)[0]

    # stage 3: personalization = blend relevance with the affinity to the user's history
    personal = np.zeros(len(cand))
    score = final
    if active:
        aff = personalization.affinity(idx, profile)
        if aff.max() > 0:
            personal = aff[cand] / aff.max()  # 1.0 = best match of the whole catalogue
            score = (1 - alpha) * minmax(final) + alpha * personal

    order = np.argsort(-score)[:k]
    results = [
        {"i": int(cand[o]), "score": float(score[o]), "personal": float(personal[o])}
        for o in order
    ]
    return results, weights, log


########### UI
def _submit():
    """Search form callback: remember the query and ask to save it in the history."""
    q = st.session_state.get("q_input", "").strip()
    if q:
        st.session_state.query = q
        st.session_state.save_pending = True  # saved once, not on every rerun
        st.session_state.refresh_profile = True


def _use_history(q):
    """Re-run an old search from the history (not saved a second time)."""
    st.session_state.q_input = q
    st.session_state.query = q
    st.session_state.refresh_profile = True


def _like(uid, name, query):
    db.add_interaction(uid, name, query)
    st.session_state.setdefault("liked", set()).add(name)


def _clear(uid):
    db.clear_history(uid)
    st.session_state.refresh_profile = True
    st.session_state.pop("liked", None)


def unique_queries(history, n):
    seen, out = set(), []
    for h in history:
        if h["query"] not in seen:
            seen.add(h["query"])
            out.append(h["query"])
        if len(out) == n:
            break
    return out


@st.cache_resource(show_spinner="Indexing the catalogue...")
def default_index():
    return Index(load_products())


def auth_screen():
    st.title("🔧 SmartHeat Plumbing Recommender")
    st.caption("Sign in to get search results personalised with your history")
    tab_in, tab_up = st.tabs(["Sign in", "Create account"])

    with tab_in:
        with st.form("signin_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            ok = st.form_submit_button("Sign in")
        if ok:
            user = db.authenticate(username, password)
            if user:
                st.session_state.user = user
                st.session_state.refresh_profile = True
                st.rerun()
            else:
                st.error("Wrong username or password.")

    with tab_up:
        with st.form("signup_form"):
            new_user = st.text_input("Choose a username")
            pw1 = st.text_input("Choose a password (6+ characters)", type="password")
            pw2 = st.text_input("Confirm the password", type="password")
            ok2 = st.form_submit_button("Create account")
        if ok2:
            if pw1 != pw2:
                st.error("The two passwords do not match.")
            else:
                done, msg = db.create_user(new_user, pw1)
                if done:
                    st.success(msg + " You can now sign in from the first tab.")
                else:
                    st.error(msg)


def main():
    st.set_page_config(
        page_title="SmartHeat Plumbing Recommender", page_icon="🔧", layout="wide"
    )
    db.init_db()

    user = st.session_state.get("user")
    if not user:
        auth_screen()
        return
    uid = user["id"]

    st.title("🔧 SmartHeat Plumbing Recommender")
    st.caption("Query expansion + re-ranking + personalization")

    with st.sidebar:
        st.markdown(f"👤 Signed in as **{user['username']}**")
        if st.button("Log out"):
            st.session_state.clear()
            st.rerun()

        st.header("Settings")
        up = st.file_uploader(
            "Your catalogue (CSV: name, category, description, price)", type="csv"
        )
        methods = st.multiselect(
            "Expansion methods",
            [
                "Synonym",
                "Related term",
                "Concept",
                "Automatic (corpus)",
                "Feedback (PRF)",
            ],
            default=["Synonym", "Related term", "Concept"],
        )
        reranker = st.selectbox(
            "Re-ranker",
            [
                "Hybrid (BM25 + TF-IDF)",
                "BM25",
                "TF-IDF",
                "Neural (sentence-transformers)",
            ],
        )
        k = st.slider("Results", 3, 10, 5)

        st.header("Personalization")
        personalize = st.toggle("Use my history", value=True)
        alpha = st.slider(
            "Strength",
            0.0,
            1.0,
            0.4,
            0.05,
            disabled=not personalize,
            help="0 = ignore my history, 1 = rank only by my history",
        )

    try:
        idx = Index(load_products(up)) if up is not None else default_index()
    except Exception as e:
        st.error(f"Could not load the catalogue: {e}")
        st.stop()

    query = st.session_state.get("query", "")

    # Interest profile, built BEFORE the current search is saved so a query never boosts itself.
    # It is kept in session_state and refreshed only when a new search starts.
    sig = (idx.N, len(idx.vocab))
    snap = st.session_state.get("profile")
    if (
        st.session_state.pop("refresh_profile", False)
        or snap is None
        or snap["sig"] != sig
    ):
        liked = [x["product_name"] for x in db.get_interactions(uid, 100)]
        queries = [h["query"] for h in db.get_history(uid, 100)]
        snap = {
            "vec": personalization.build_profile(idx, tokenize, queries, liked),
            "cats": personalization.top_categories(idx, liked),
            "sig": sig,
        }
        st.session_state.profile = snap

    if st.session_state.pop("save_pending", False) and query:
        db.add_search(uid, query)
    history = db.get_history(uid, 100)

    tab_search, tab_hist = st.tabs(["🔍 Search", "🕘 My history"])

    with tab_search:
        with st.form("search_form"):
            st.text_input(
                "What are you looking for?",
                key="q_input",
                placeholder="e.g. fix dripping faucet, heating for bathroom, hot water",
            )
            st.form_submit_button("Search", on_click=_submit)

        recent = unique_queries(history, 5)
        if recent:
            st.caption("Recent searches")
            for col, q in zip(st.columns(len(recent)), recent):
                col.button(
                    q,
                    key=f"recent_{q}",
                    on_click=_use_history,
                    args=(q,),
                    width="stretch",
                )

        if not query:
            st.info("Type a query above to get recommendations.")
        else:
            results, weights, log = search(
                query,
                methods,
                reranker,
                k,
                idx,
                profile=snap["vec"] if personalize else None,
                alpha=alpha if personalize else 0.0,
            )
            left, right = st.columns([2, 1])
            with right:
                st.subheader("Query analysis")
                st.write(
                    "**Original terms:**",
                    ", ".join(t for t, w in weights.items() if w == 1.0) or "-",
                )
                if log:
                    st.dataframe(pd.DataFrame(log), hide_index=True, width="stretch")
                else:
                    st.info("No expansion terms added.")
            with left:
                st.subheader("Recommendations")
                if personalize:
                    terms = personalization.top_terms(idx, snap["vec"])
                    if terms:
                        st.caption(
                            "🎯 Personalized with your interests: " + ", ".join(terms)
                        )
                    else:
                        st.caption(
                            "🎯 Personalization is on, but your history is empty. "
                            "Search a few times and press 👍 on products you like."
                        )
                if not results:
                    st.warning("No matching products.")
                liked_now = st.session_state.get("liked", set())
                for r in results:
                    row = idx.df.iloc[r["i"]]
                    with st.container(border=True):
                        st.markdown(
                            f"**{row['name']}**  ·  `{row['category']}`  ·  **${row['price']}**"
                            + (
                                f"  ·  {row['brand']}"
                                if "brand" in row and pd.notna(row["brand"])
                                else ""
                            )
                        )
                        st.write(row["description"])
                        c1, c2 = st.columns([3, 1])
                        caption = f"Relevance score: {r['score']:.3f}"
                        if r["personal"] > 0.05:
                            caption += f"  ·  🎯 interest match {r['personal']:.0%}"
                        c1.caption(caption)
                        c2.button(
                            "✅ Noted" if row["name"] in liked_now else "👍 Interested",
                            key=f"like_{r['i']}",
                            on_click=_like,
                            args=(uid, row["name"], query),
                            disabled=row["name"] in liked_now,
                            width="stretch",
                        )

    with tab_hist:
        inter = db.get_interactions(uid, 100)
        st.subheader("Your profile")
        terms = personalization.top_terms(idx, snap["vec"], 8)
        st.write(
            "**Interest keywords:**", ", ".join(terms) if terms else "- (nothing yet)"
        )
        st.write(
            "**Favourite categories:**",
            ", ".join(snap["cats"]) if snap["cats"] else "- (nothing yet)",
        )

        st.subheader("Searches")
        if history:
            st.dataframe(
                pd.DataFrame(history).rename(
                    columns={"query": "Query", "created_at": "Date (UTC)"}
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("No searches yet.")

        st.subheader("Products you were interested in")
        if inter:
            st.dataframe(
                pd.DataFrame(inter).rename(
                    columns={
                        "product_name": "Product",
                        "query": "Found with query",
                        "created_at": "Date (UTC)",
                    }
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("Press 👍 Interested on a product to add it here.")

        st.button("🗑 Clear my history", on_click=_clear, args=(uid,))


if __name__ == "__main__":
    main()
