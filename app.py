"""Heating & Plumbing product recommender
Query Expansion (synonym / related / concept / feedback / automatic) + Re-ranking (BM25, TF-IDF, hybrid, neural)
Run:  streamlit run app.py
"""

import math
import re
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. DATASET  (replace with your own CSV: name, category, description, price)
PRODUCTS = [
    (
        "Condensing Combi Boiler 30kW",
        "Boilers",
        "Wall-mounted gas combi boiler providing central heating and instant hot water, high efficiency with flue kit",
        1250,
    ),
    (
        "Oil System Boiler 24kW",
        "Boilers",
        "Floor standing oil fired boiler for heating and hot water cylinder",
        1890,
    ),
    (
        "Electric Water Heater 80L",
        "Water Heating",
        "Storage tank water heater with thermostat for hot water supply",
        420,
    ),
    (
        "Panel Radiator 600x1000",
        "Radiators",
        "Double panel steel radiator for room heating, includes wall brackets",
        85,
    ),
    (
        "Towel Radiator Chrome",
        "Radiators",
        "Bathroom heated towel rail radiator, chrome finish",
        120,
    ),
    (
        "Thermostatic Radiator Valve",
        "Valves",
        "TRV for controlling radiator temperature, angled chrome",
        18,
    ),
    ("Radiator Bleed Key", "Accessories", "Key to bleed trapped air from radiators", 3),
    (
        "Smart Room Thermostat",
        "Controls",
        "WiFi programmable thermostat controller for heating system",
        95,
    ),
    (
        "Central Heating Circulator Pump",
        "Pumps",
        "Variable speed circulation pump for heating circuits",
        160,
    ),
    (
        "Underfloor Heating Kit 10m2",
        "Underfloor Heating",
        "Electric underfloor heating mat with thermostat for bathroom or kitchen floors",
        210,
    ),
    (
        "Expansion Vessel 18L",
        "Heating Parts",
        "Pressure expansion vessel for sealed heating systems",
        45,
    ),
    (
        "Magnetic System Filter",
        "Heating Parts",
        "Magnetic filter removes sludge and debris from heating water, protects boiler",
        78,
    ),
    (
        "Copper Pipe 15mm 3m",
        "Pipes",
        "Copper tube for water supply and heating pipework",
        22,
    ),
    (
        "PEX Pipe 16mm 50m",
        "Pipes",
        "Flexible plastic piping for hot and cold water plumbing",
        64,
    ),
    ("PVC Waste Pipe 40mm", "Pipes", "Waste drain pipe for sinks and basins", 9),
    (
        "Compression Elbow 15mm",
        "Fittings",
        "Brass elbow fitting joins copper pipes without soldering",
        2.5,
    ),
    ("Pipe Coupling 22mm", "Fittings", "Straight coupler connecting two pipes", 3),
    (
        "Ball Valve 1/2 inch",
        "Valves",
        "Quarter turn brass shutoff valve stopcock for water isolation",
        8,
    ),
    (
        "Kitchen Mixer Tap Chrome",
        "Taps",
        "Single lever kitchen faucet mixer with swivel spout",
        75,
    ),
    ("Basin Tap Waterfall", "Taps", "Bathroom basin faucet with ceramic cartridge", 55),
    (
        "Tap Cartridge Replacement",
        "Spare Parts",
        "Ceramic cartridge to fix dripping leaking taps",
        12,
    ),
    (
        "Thermostatic Shower Mixer",
        "Showers",
        "Exposed shower valve with temperature control, includes rain head",
        140,
    ),
    (
        "Close Coupled Toilet",
        "Sanitary",
        "Ceramic WC toilet with soft close seat and cistern",
        160,
    ),
    (
        "Pedestal Wash Basin",
        "Sanitary",
        "Ceramic sink basin with pedestal for bathroom",
        70,
    ),
    (
        "PTFE Thread Seal Tape",
        "Consumables",
        "Tape to seal threaded joints and stop leaks",
        2,
    ),
    (
        "Pipe Repair Clamp",
        "Consumables",
        "Emergency clamp to stop pipe leak and burst repair",
        15,
    ),
    (
        "Pipe Insulation Foam 2m",
        "Consumables",
        "Foam lagging to prevent frozen pipes and heat loss",
        4,
    ),
    (
        "Silicone Sealant",
        "Consumables",
        "Waterproof sealant for sinks, baths and showers",
        7,
    ),
]


def load_products(uploaded=None) -> pd.DataFrame:
    import os

    if uploaded is not None:
        df = pd.read_csv(uploaded)
    elif os.path.exists("data/products.csv"):
        df = pd.read_csv("data/products.csv")
    else:
        df = pd.DataFrame(
            PRODUCTS, columns=["name", "category", "description", "price"]
        )
    for col in ("name", "category", "description"):
        df[col] = df[col].fillna("").astype(str)
    return df


# 2. KNOWLEDGE BASES for expansion (domain dictionaries)
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


# 3. STEP 1 - IDENTIFY QUERY TERMS (tokenize, stop-words, light stemming)
def stem(w: str) -> str:
    for suf in ("ing", "ed", "s"):
        if w.endswith(suf) and not w.endswith("ss") and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


# 4. INDEX: BM25 + TF-IDF over the product catalogue


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


# 5. STEPS 2 & 3 - EXPAND QUERY TERMS
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


# 6. STEP 4 - RE-RANKING
@st.cache_resource(show_spinner="Loading neural model...")
def load_neural():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def minmax(x):
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng else np.zeros_like(x)


def search(query, methods, reranker, k, idx: Index, n_candidates=12):
    weights, log = expand_query(query, methods, idx)
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

    order = np.argsort(-final)[:k]
    results = [(int(cand[o]), float(final[o])) for o in order]
    return results, weights, log


# 7. UI
def main():
    st.set_page_config(
        page_title="Heating & Plumbing Finder", page_icon="🔧", layout="wide"
    )
    st.title("🔧 SmartHeat Plumbing Recommender")
    st.caption("Query expansion + re-ranking")

    with st.sidebar:
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

    idx = Index(load_products(up))
    query = st.text_input(
        "What are you looking for?",
        placeholder="e.g. fix dripping faucet, heating for bathroom, hot water",
    )

    if query:
        results, weights, log = search(query, methods, reranker, k, idx)
        left, right = st.columns([2, 1])
        with right:
            st.subheader("Query analysis")
            st.write(
                "**Original terms:**",
                ", ".join(t for t, w in weights.items() if w == 1.0) or "-",
            )
            if log:
                st.dataframe(
                    pd.DataFrame(log), hide_index=True, use_container_width=True
                )
            else:
                st.info("No expansion terms added.")
        with left:
            st.subheader("Recommendations")
            if not results:
                st.warning("No matching products.")
            for i, score in results:
                r = idx.df.iloc[i]
                with st.container(border=True):
                    st.markdown(
                        f"**{r['name']}**  ·  `{r['category']}`  ·  **${r['price']}**"
                        + (
                            f"  ·  {r['brand']}"
                            if "brand" in r and pd.notna(r["brand"])
                            else ""
                        )
                    )
                    st.write(r["description"])
                    st.caption(f"Relevance score: {score:.3f}")


if __name__ == "__main__":
    main()
