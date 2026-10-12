"""SQLite-backed authentication + prediction history for the dashboard."""
import hashlib
import hmac
import re
import secrets
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "dashboard.db"
DEFAULT_ADMIN_USER = "admin"
DEFAULT_ADMIN_PASS = "Admin@123"   # change after first login


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000).hex()


def init_db():
    with _conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP)"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS predictions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                text TEXT,
                language TEXT,
                label TEXT,
                probability REAL,
                source TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP)"""
        )
        n = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if n == 0:
        create_user(DEFAULT_ADMIN_USER, "admin@local", DEFAULT_ADMIN_PASS, role="admin")


def create_user(username, email, password, role="user"):
    username = username.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,30}", username):
        return False, "Username must be 3-30 characters (letters, digits, _ . -)."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    salt = secrets.token_hex(16)
    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO users(username,email,password_hash,salt,role) VALUES(?,?,?,?,?)",
                (username, email.strip(), _hash(password, salt), salt, role),
            )
        return True, "Account created."
    except sqlite3.IntegrityError:
        return False, "Username already exists."


def verify_user(username, password):
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE username=?", (username.strip(),)).fetchone()
    if not row:
        return None
    if hmac.compare_digest(_hash(password, row["salt"]), row["password_hash"]):
        return {"username": row["username"], "role": row["role"], "email": row["email"]}
    return None


def change_password(username, new_password):
    if len(new_password) < 6:
        return False, "Password must be at least 6 characters."
    salt = secrets.token_hex(16)
    with _conn() as c:
        c.execute("UPDATE users SET password_hash=?, salt=? WHERE username=?",
                  (_hash(new_password, salt), salt, username))
    return True, "Password updated."


def list_users():
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT id,username,email,role,created_at FROM users ORDER BY id")]


def delete_user(username):
    if username == DEFAULT_ADMIN_USER:
        return False
    with _conn() as c:
        c.execute("DELETE FROM users WHERE username=?", (username,))
    return True


def set_role(username, role):
    with _conn() as c:
        c.execute("UPDATE users SET role=? WHERE username=?", (role, username))


def log_prediction(username, text, language, label, prob, source):
    with _conn() as c:
        c.execute(
            "INSERT INTO predictions(username,text,language,label,probability,source) VALUES(?,?,?,?,?,?)",
            (username, text[:1000], language, label, float(prob), source),
        )


def get_predictions(username=None, limit=1000):
    q = "SELECT id,username,text,language,label,probability,source,created_at FROM predictions"
    args = []
    if username:
        q += " WHERE username=?"
        args.append(username)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    with _conn() as c:
        return [dict(r) for r in c.execute(q, args)]


def clear_predictions(username=None):
    with _conn() as c:
        if username:
            c.execute("DELETE FROM predictions WHERE username=?", (username,))
        else:
            c.execute("DELETE FROM predictions")
