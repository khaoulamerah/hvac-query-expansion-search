"""SQLite storage for SmartHeat: user accounts, search history and product interactions.

Only the standard library is used (sqlite3, hashlib), so no extra dependency is needed.
Passwords are never stored in clear text: PBKDF2-HMAC-SHA256 with a random salt per user.
"""

import hashlib
import hmac
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime

DB_PATH = os.path.join("data", "users.db")
PBKDF2_ROUNDS = 200_000
USERNAME_RE = re.compile(r"[A-Za-z0-9_.-]{3,30}")


@contextmanager
def connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    """Create the tables if they do not exist yet (safe to call on every run)."""
    with connect() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                salt          TEXT NOT NULL,
                created_at    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS search_history (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                query      TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS interactions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                product_name TEXT NOT NULL,
                query        TEXT,
                created_at   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_history_user ON search_history(user_id, id);
            CREATE INDEX IF NOT EXISTS idx_inter_user ON interactions(user_id, id);
            """
        )


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")


def _hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS
    ).hex()



# Accounts
def create_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip()
    if not USERNAME_RE.fullmatch(username):
        return False, "Username must be 3-30 characters: letters, digits, _ . -"
    if not 6 <= len(password) <= 128:
        return False, "Password must be between 6 and 128 characters."
    salt = os.urandom(16)
    try:
        with connect() as con:
            con.execute(
                "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
                (username, _hash(password, salt), salt.hex(), _now()),
            )
    except sqlite3.IntegrityError:
        return False, "This username is already taken."
    return True, "Account created."


def authenticate(username: str, password: str) -> dict | None:
    with connect() as con:
        row = con.execute(
            "SELECT id, username, password_hash, salt FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    if row is None:
        return None
    candidate = _hash(password, bytes.fromhex(row["salt"]))
    if hmac.compare_digest(candidate, row["password_hash"]):
        return {"id": row["id"], "username": row["username"]}
    return None



# Search history
def add_search(user_id: int, query: str) -> None:
    with connect() as con:
        con.execute(
            "INSERT INTO search_history (user_id, query, created_at) VALUES (?, ?, ?)",
            (user_id, query, _now()),
        )


def get_history(user_id: int, limit: int = 100) -> list[dict]:
    with connect() as con:
        rows = con.execute(
            "SELECT query, created_at FROM search_history WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]



# Product interactions ("I'm interested" clicks)
def add_interaction(user_id: int, product_name: str, query: str | None) -> None:
    with connect() as con:
        con.execute(
            "INSERT INTO interactions (user_id, product_name, query, created_at) VALUES (?, ?, ?, ?)",
            (user_id, product_name, query, _now()),
        )


def get_interactions(user_id: int, limit: int = 100) -> list[dict]:
    """Products the user marked as interesting, newest first."""
    with connect() as con:
        rows = con.execute(
            "SELECT product_name, query, created_at FROM interactions WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def clear_history(user_id: int) -> None:
    with connect() as con:
        con.execute("DELETE FROM search_history WHERE user_id = ?", (user_id,))
        con.execute("DELETE FROM interactions WHERE user_id = ?", (user_id,))
