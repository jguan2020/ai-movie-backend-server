import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import psycopg2
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR.parent.parent / ".env")

MOVIE_DATABASE_URL = os.getenv("MOVIE_DATABASE_URL")
USER_DATABASE_URL = os.getenv("USER_DATABASE_URL")
FAVORITES_DATABASE_URL = os.getenv("FAVORITES_DATABASE_URL")
IS_PREMIUM_DATABASE_URL = os.getenv("IS_PREMIUM_DATABASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
TIER_ONE_STRIPE_ID = os.getenv("TIER_ONE_STRIPE_ID", "")
STRIPE_RETURN_URL = os.getenv("STRIPE_RETURN_URL", "")
JWT_SECRET = os.getenv("JWT_SECRET", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CANONICAL_TABLE = os.getenv("CANONICAL_TABLE", "top1000")
IMG_BASE = "https://image.tmdb.org/t/p/w342"
JWT_TTL_SECONDS = int(os.getenv("JWT_TTL_SECONDS", "604800"))
FREE_FAVORITES_LIMIT = 10
ACTIVE_SUBSCRIPTION_STATUSES = {"active", "trialing"}

LANG_NAMES = {
    "af": "Afrikaans",
    "am": "Amharic",
    "ar": "Arabic",
    "as": "Assamese",
    "ay": "Aymara",
    "az": "Azerbaijani",
    "bg": "Bulgarian",
    "bm": "Bambara",
    "bn": "Bengali",
    "bo": "Tibetan",
    "bs": "Bosnian",
    "ca": "Catalan",
    "cn": "Cantonese",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "dz": "Dzongkha",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fo": "Faroese",
    "fr": "French",
    "fy": "Western Frisian",
    "ga": "Irish",
    "gl": "Galician",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "ht": "Haitian Creole",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "ig": "Igbo",
    "is": "Icelandic",
    "it": "Italian",
    "iu": "Inuktitut",
    "ja": "Japanese",
    "jv": "Javanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "ks": "Kashmiri",
    "ku": "Kurdish",
    "ky": "Kyrgyz",
    "ln": "Lingala",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mo": "Moldovan",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Burmese",
    "nb": "Norwegian Bokmal",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "or": "Odia",
    "pa": "Punjabi",
    "pl": "Polish",
    "ps": "Pashto",
    "pt": "Portuguese",
    "qu": "Quechua",
    "ro": "Romanian",
    "ru": "Russian",
    "sa": "Sanskrit",
    "se": "Northern Sami",
    "sh": "Serbo-Croatian",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sn": "Shona",
    "sq": "Albanian",
    "sr": "Serbian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tn": "Tswana",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "wo": "Wolof",
    "xx": "Unknown",
    "yi": "Yiddish",
    "zh": "Chinese",
    "zu": "Zulu",
}

COMMON_LANG_ORDER = [
    "en",
    "zh",
    "cn",
    "es",
    "hi",
    "ar",
    "pt",
    "ru",
    "ja",
    "de",
    "fr",
    "ko",
    "it",
]

SAMPLE_FEATURED = [
    {
        "title": "Avatar: Fire and Ash",
        "release_date": "2025-12-17",
        "rating": "PG-13",
        "runtime": 198,
        "popularity": 490.0904,
        "genres": [
            "Science Fiction",
            "Adventure",
            "Fantasy",
        ],
        "poster_path": "/g96wHxU7EnoIFwemb2RgohIXrgW.jpg",
        "keywords_topics": [
            "aliens",
            "interstellar war",
            "indigenous culture",
            "military dictatorship",
            "power struggle",
            "family",
        ],
        "overview": "In the wake of the devastating war against the RDA and the loss of their eldest son, Jake Sully and Neytiri face a new threat on Pandora: the Ash People, a violent and power-hungry Na'vi tribe led by the ruthless Varang. Jake's family must fight for their survival and the future of Pandora in a conflict that pushes them to their emotional and physical limits.",
    },
    {
        "title": "B\u0101hubali: The Epic",
        "release_date": "2025-10-29",
        "rating": "NR",
        "runtime": 224,
        "popularity": 440.7241,
        "genres": [
            "Action",
            "Drama",
        ],
        "poster_path": "/4sLSorDKKDN944kWngxgQlpdDeg.jpg",
        "keywords_topics": [
            "high fantasy",
            "medieval",
            "monarchy",
            "whodunit",
            "sibling rivalry",
            "revenge",
        ],
        "overview": "When a mysterious child is found by a tribal couple near a roaring waterfall, they raise him as their own. As he grows, Sivudu is drawn to the world beyond the cliffs, where he discovers the ancient kingdom of Mahishmati, ruled by a cruel tyrant, haunted by rebellion, and bound to his past. What begins as a quest for love soon unravels a legacy of betrayal, sacrifice, and a forgotten prince.",
    },
    {
        "title": "Zootopia 2",
        "release_date": "2025-11-26",
        "rating": "PG",
        "runtime": 107,
        "popularity": 380.0296,
        "genres": [
            "Animation",
            "Comedy",
            "Adventure",
            "Family",
            "Mystery",
        ],
        "poster_path": "/bjUWGw0Ao0qVWxagN3VCwBJHVo6.jpg",
        "keywords_topics": [
            "animation",
            "buddy cop",
            "animals/talking animals",
            "crime",
            "illustration",
        ],
        "overview": "After cracking the biggest case in Zootopia's history, rookie cops Judy Hopps and Nick Wilde find themselves on the twisting trail of a great mystery when Gary De'Snake arrives and turns the animal metropolis upside down. To crack the case, Judy and Nick must go undercover to unexpected new parts of town, where their growing partnership is tested like never before.",
    },
    {
        "title": "Demon Slayer: Kimetsu no Yaiba Infinity Castle",
        "release_date": "2025-07-18",
        "rating": "R",
        "runtime": 156,
        "popularity": 221.0241,
        "genres": [
            "Animation",
            "Action",
            "Fantasy",
        ],
        "poster_path": "/fWVSwgjpT2D78VUh6X8UBd2rorW.jpg",
        "keywords_topics": [
            "demons",
            "dark fantasy",
            "supernatural",
            "action",
            "battle",
            "animation",
        ],
        "overview": "The Demon Slayer Corps are drawn into the Infinity Castle, where Tanjiro, Nezuko, and the Hashira face terrifying Upper Rank demons in a desperate fight as the final battle against Muzan Kibutsuji begins.",
    },
    {
        "title": "Avatar: The Way of Water",
        "release_date": "2022-12-14",
        "rating": "PG-13",
        "runtime": 192,
        "popularity": 123.8467,
        "genres": [
            "Action",
            "Adventure",
            "Science Fiction",
        ],
        "poster_path": "/t6HIqrRAclMCA60NsSmeqe9RmNV.jpg",
        "keywords_topics": [
            "aliens",
            "interstellar war",
            "underwater",
            "native americans",
            "family",
            "colonialism",
        ],
        "overview": "Set more than a decade after the events of the first film, learn the story of the Sully family (Jake, Neytiri, and their kids), the trouble that follows them, the lengths they go to keep each other safe, the battles they fight to stay alive, and the tragedies they endure.",
    },
    {
        "title": "The Housemaid",
        "release_date": "2025-12-18",
        "rating": "R",
        "runtime": 131,
        "popularity": 123.4737,
        "genres": [
            "Mystery",
            "Thriller",
        ],
        "poster_path": "/cWsBscZzwu5brg9YjNkGewRUvJX.jpg",
        "keywords_topics": [
            "psychological thriller",
            "domestic violence",
            "servants",
            "family legacy",
            "conspiracy theories",
            "trauma",
        ],
        "overview": "Trying to escape her past, Millie Calloway accepts a job as a live-in housemaid for the wealthy Nina and Andrew Winchester. But what begins as a dream job quickly unravels into something far more dangerous\u2014a sexy, seductive game of secrets, scandal, and power.",
    },
    {
        "title": "The Shadow's Edge",
        "release_date": "2025-08-16",
        "rating": "NR",
        "runtime": 142,
        "popularity": 123.058,
        "genres": [
            "Action",
            "Crime",
            "Thriller",
        ],
        "poster_path": "/e0RU6KpdnrqFxDKlI3NOqN8nHL6.jpg",
        "keywords_topics": [
            "heist",
            "surveillance",
            "crime",
            "investigation",
            "police",
        ],
        "overview": "Macau Police brings the tracking expert police officer out of retirement to help catch a dangerous group of professional thieves.",
    },
    {
        "title": "Predator: Badlands",
        "release_date": "2025-11-05",
        "rating": "PG-13",
        "runtime": 107,
        "popularity": 122.7678,
        "genres": [
            "Action",
            "Science Fiction",
            "Adventure",
        ],
        "poster_path": "/ef2QSeBkrYhAdfsWGXmp0lvH0T1.jpg",
        "keywords_topics": [
            "aliens",
            "space",
            "dragons",
            "survival",
            "agriculture",
            "coming-of-age",
            "corporate corruption",
        ],
        "overview": "Cast out from his clan, a young Predator finds an unlikely ally in a damaged android and embarks on a treacherous journey in search of the ultimate adversary.",
    },
]

app = FastAPI()

allowed_origins = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "").split(",") if origin.strip()]
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


class SearchRequest(BaseModel):
    overview_query: str = ""
    language: str = "Any"


class AuthRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str
    confirm_password: str


class FavoriteRequest(BaseModel):
    title: str
    release_date: Optional[str] = ""
    poster_path: Optional[str] = ""
    genres: Optional[str] = ""


class UpdateEmailRequest(BaseModel):
    new_email: str
    current_password: str


class UpdatePasswordRequest(BaseModel):
    current_password: str
    new_password: str
    confirm_password: str


class SubscribeRequest(BaseModel):
    return_url: str = ""


def format_language(code: str) -> str:
    return LANG_NAMES.get(code, code)


def order_languages(languages: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for code in COMMON_LANG_ORDER:
        if code in languages and code not in seen:
            ordered.append(code)
            seen.add(code)
    remaining = [code for code in languages if code not in seen]
    remaining.sort(key=lambda code: format_language(code).lower())
    ordered.extend(remaining)
    return ordered


def get_conn():
    if not MOVIE_DATABASE_URL:
        raise RuntimeError("Movie database is not configured.")
    return psycopg2.connect(MOVIE_DATABASE_URL, sslmode="require")


def get_user_conn():
    if not USER_DATABASE_URL:
        raise RuntimeError("User database is not configured.")
    return psycopg2.connect(USER_DATABASE_URL, sslmode="require")


def get_favorites_conn():
    if not FAVORITES_DATABASE_URL:
        raise RuntimeError("Favorites database is not configured.")
    return psycopg2.connect(FAVORITES_DATABASE_URL, sslmode="require")


def get_premium_conn():
    if not IS_PREMIUM_DATABASE_URL:
        raise RuntimeError("Premium database is not configured.")
    return psycopg2.connect(IS_PREMIUM_DATABASE_URL, sslmode="require")


def normalize_email(email: str) -> str:
    return email.strip().lower()


def base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def base64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    iterations = 200_000
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${base64url_encode(salt)}${base64url_encode(derived)}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        alg, iter_str, salt_b64, hash_b64 = stored_hash.split("$", 3)
        if alg != "pbkdf2_sha256":
            return False
        iterations = int(iter_str)
        salt = base64url_decode(salt_b64)
        expected = base64url_decode(hash_b64)
        derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(derived, expected)
    except Exception:
        return False


def create_token(email: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": email,
        "iat": int(time.time()),
        "exp": int(time.time()) + JWT_TTL_SECONDS,
    }
    header_b64 = base64url_encode(json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    payload_b64 = base64url_encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    signature = hmac.new(JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{header_b64}.{payload_b64}.{base64url_encode(signature)}"


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        header_b64, payload_b64, signature_b64 = token.split(".", 2)
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        signature = base64url_decode(signature_b64)
        expected = hmac.new(JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            return None
        payload = json.loads(base64url_decode(payload_b64).decode("utf-8"))
        exp = payload.get("exp")
        if exp and int(exp) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


def get_optional_user(authorization: Optional[str]) -> Optional[str]:
    if not JWT_SECRET:
        return None
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    payload = decode_token(parts[1])
    if not payload:
        return None
    return payload.get("sub")


def get_required_user(authorization: Optional[str]) -> str:
    email = get_optional_user(authorization)
    if not email:
        raise HTTPException(status_code=401, detail="not_authenticated")
    return email


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    with get_user_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT id, email, password_hash FROM app_users WHERE email = %s;", (email,))
        return cur.fetchone()


def create_user(email: str, password: str) -> bool:
    password_hash = hash_password(password)
    try:
        with get_user_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO app_users (email, password_hash) VALUES (%s, %s);",
                (email, password_hash),
            )
        return True
    except psycopg2.IntegrityError:
        return False


def update_user_email(old_email: str, new_email: str) -> bool:
    try:
        with get_user_conn() as conn, conn.cursor() as cur:
            cur.execute("UPDATE app_users SET email = %s WHERE email = %s;", (new_email, old_email))
            return cur.rowcount > 0
    except psycopg2.IntegrityError:
        return False


def update_user_password(email: str, new_password: str) -> bool:
    password_hash = hash_password(new_password)
    with get_user_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE app_users SET password_hash = %s WHERE email = %s;",
            (password_hash, email),
        )
        return cur.rowcount > 0


def get_premium_record(email: str) -> Optional[Dict[str, Any]]:
    if not IS_PREMIUM_DATABASE_URL:
        return None
    try:
        with get_premium_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT is_premium, stripe_subscription_id, cancel_at_period_end, current_period_end
                FROM premium_users
                WHERE user_email = %s;
                """,
                (email,),
            )
            return cur.fetchone()
    except Exception:
        return None


def is_premium_active(record: Optional[Dict[str, Any]]) -> bool:
    if not record:
        return False
    if record.get("is_premium"):
        return True
    if record.get("cancel_at_period_end") and record.get("current_period_end"):
        try:
            return record["current_period_end"] > datetime.now(timezone.utc)
        except Exception:
            return False
    return False


def get_user_is_premium(email: str) -> bool:
    record = get_premium_record(email)
    return is_premium_active(record)


def set_user_premium(
    email: str,
    is_premium: bool,
    subscription_id: Optional[str] = None,
    period_end: Optional[datetime] = None,
    cancel_at_period_end: Optional[bool] = None,
    clear_subscription: bool = False,
) -> None:
    if not IS_PREMIUM_DATABASE_URL:
        return
    try:
        with get_premium_conn() as conn, conn.cursor() as cur:
            if clear_subscription:
                cur.execute(
                    """
                    INSERT INTO premium_users (user_email, is_premium, stripe_subscription_id, cancel_at_period_end, current_period_end)
                    VALUES (%s, %s, NULL, FALSE, NULL)
                    ON CONFLICT (user_email) DO UPDATE SET
                        is_premium = EXCLUDED.is_premium,
                        stripe_subscription_id = NULL,
                        cancel_at_period_end = FALSE,
                        current_period_end = NULL;
                    """,
                    (email, is_premium),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO premium_users (user_email, is_premium, stripe_subscription_id, cancel_at_period_end, current_period_end)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (user_email) DO UPDATE SET
                        is_premium = EXCLUDED.is_premium,
                        stripe_subscription_id = COALESCE(EXCLUDED.stripe_subscription_id, premium_users.stripe_subscription_id),
                        cancel_at_period_end = COALESCE(EXCLUDED.cancel_at_period_end, premium_users.cancel_at_period_end),
                        current_period_end = COALESCE(EXCLUDED.current_period_end, premium_users.current_period_end);
                    """,
                    (email, is_premium, subscription_id, cancel_at_period_end, period_end),
                )
    except Exception:
        return


def update_premium_by_subscription_id(
    subscription_id: str,
    is_premium: bool,
    period_end: Optional[datetime] = None,
    cancel_at_period_end: Optional[bool] = None,
    clear_subscription: bool = False,
) -> None:
    if not IS_PREMIUM_DATABASE_URL:
        return
    try:
        with get_premium_conn() as conn, conn.cursor() as cur:
            if clear_subscription:
                cur.execute(
                    """
                    UPDATE premium_users
                    SET is_premium = %s,
                        stripe_subscription_id = NULL,
                        cancel_at_period_end = FALSE,
                        current_period_end = NULL
                    WHERE stripe_subscription_id = %s;
                    """,
                    (is_premium, subscription_id),
                )
            else:
                cur.execute(
                    """
                    UPDATE premium_users
                    SET is_premium = %s,
                        cancel_at_period_end = COALESCE(%s, cancel_at_period_end),
                        current_period_end = COALESCE(%s, current_period_end)
                    WHERE stripe_subscription_id = %s;
                    """,
                    (is_premium, cancel_at_period_end, period_end, subscription_id),
                )
    except Exception:
        return


def build_favorite_key(title: str, release_date: str) -> str:
    safe_title = (title or "").strip()
    safe_date = (release_date or "").strip()
    return f"{safe_title}|{safe_date}"


def get_favorites_for_user(email: str) -> List[Dict[str, Any]]:
    with get_favorites_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT movie_title, release_date, poster_path, genres
            FROM favorites
            WHERE user_email = %s
            ORDER BY created_at DESC;
            """,
            (email,),
        )
        return cur.fetchall()


def update_favorites_email(old_email: str, new_email: str) -> None:
    if not FAVORITES_DATABASE_URL:
        return
    with get_favorites_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE favorites SET user_email = %s WHERE user_email = %s;",
            (new_email, old_email),
        )


def update_premium_email(old_email: str, new_email: str) -> None:
    if not IS_PREMIUM_DATABASE_URL:
        return
    with get_premium_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE premium_users SET user_email = %s WHERE user_email = %s;",
            (new_email, old_email),
        )


def favorite_exists_for_user(email: str, title: str, release_date: str) -> bool:
    release_date = release_date or ""
    with get_favorites_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM favorites
            WHERE user_email = %s AND movie_title = %s AND release_date = %s
            LIMIT 1;
            """,
            (email, title, release_date),
        )
        return cur.fetchone() is not None


def count_favorites_for_user(email: str) -> int:
    with get_favorites_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM favorites WHERE user_email = %s;", (email,))
        row = cur.fetchone()
        return int(row[0]) if row else 0


def remove_favorite_for_user(email: str, title: str, release_date: str) -> None:
    release_date = release_date or ""
    with get_favorites_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM favorites
            WHERE user_email = %s AND movie_title = %s AND release_date = %s;
            """,
            (email, title, release_date),
        )


def add_favorite_for_user(
    email: str,
    title: str,
    release_date: str,
    poster_path: str,
    genres: str,
) -> None:
    release_date = release_date or ""
    with get_favorites_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO favorites (user_email, movie_title, release_date, poster_path, genres)
            VALUES (%s, %s, %s, %s, %s);
            """,
            (email, title, release_date, poster_path or "", genres or ""),
        )


@lru_cache(maxsize=1)
def get_embedder():
    if not LLM_API_KEY:
        raise RuntimeError("Embeddings are not configured.")
    return OpenAI(api_key=LLM_API_KEY)


@lru_cache(maxsize=1)
def load_canonical_tags() -> List[str]:
    tags: List[str] = []
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT tag FROM {CANONICAL_TABLE} ORDER BY tag;")
            rows = cur.fetchall()
    except Exception as exc:
        raise RuntimeError("Canonical tag table not available.") from exc
    for row in rows:
        tag = row[0]
        if tag:
            tags.append(tag)
    return tags


@lru_cache(maxsize=1)
def embed_canonicals():
    tags = load_canonical_tags()
    client = get_embedder()
    resp = client.embeddings.create(model=EMBED_MODEL, input=tags)
    vectors = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return tags, vectors


def map_query_to_tags(query: str) -> List[str]:
    tokens = [t.strip() for t in query.split(",") if t.strip()]
    if not tokens:
        return []
    if not LLM_API_KEY:
        return tokens
    try:
        tags, canon_vectors = embed_canonicals()
        client = get_embedder()
        resp = client.embeddings.create(model=EMBED_MODEL, input=tokens)
        token_vectors = np.array([d.embedding for d in resp.data], dtype=np.float32)
        norms = np.linalg.norm(canon_vectors, axis=1) + 1e-8

        chosen: List[str] = []
        seen = set()
        for vec in token_vectors:
            qnorm = np.linalg.norm(vec) + 1e-8
            sims = (canon_vectors @ vec) / (norms * qnorm)
            idx = int(sims.argmax())
            tag = tags[idx]
            if tag not in seen:
                seen.add(tag)
                chosen.append(tag)
        return chosen
    except Exception:
        return tokens


def fetch_movies(language: Optional[str], tags: List[str], limit: int = 50) -> List[Dict[str, Any]]:
    clauses = []
    params: List[Any] = []
    has_tags = bool(tags)
    match_expr = "0"
    if has_tags:
        match_expr = (
            "cardinality((SELECT array(SELECT unnest(keywords_topics_canonical) INTERSECT "
            "SELECT unnest(%s::text[]))))"
        )
        params.append(tags)

    if language and language != "Any":
        clauses.append("language = %s")
        params.append(language)

    if has_tags:
        clauses.append("keywords_topics_canonical && %s::text[]")
        params.append(tags)

    where = "WHERE " + " AND ".join(clauses) if clauses else ""

    sql = f"""
        SELECT title, release_date, rating, runtime, popularity, genres, poster_path, overview,
               keywords_topics_canonical AS keywords_topics,
               {match_expr} AS match_count
        FROM movies
        {where}
        ORDER BY match_count DESC, popularity DESC NULLS LAST
        LIMIT %s;
    """
    params.append(limit)

    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def prepare_card(row: Dict[str, Any], chosen_tags: List[str]) -> Dict[str, Any]:
    title_text = row.get("title", "") or ""
    rating_value = row.get("rating")
    runtime_value = row.get("runtime")
    rating = str(rating_value) if rating_value else ""
    runtime = str(runtime_value) if runtime_value else ""
    meta_parts = []
    if row.get("release_date"):
        meta_parts.append(str(row["release_date"]))
    if rating:
        meta_parts.append(rating)
    if runtime:
        meta_parts.append(f"{runtime} min")
    meta = " - ".join(meta_parts)
    genres = ", ".join(row.get("genres") or [])
    keywords = row.get("keywords_topics") or []
    matched = ""
    if chosen_tags:
        matched_set = sorted(set(keywords) & set(chosen_tags))
        matched = ", ".join(matched_set)
    poster_path = row.get("poster_path")
    poster_url = f"{IMG_BASE}{poster_path}" if poster_path else ""
    release_date = str(row.get("release_date") or "")
    return {
        "title": title_text,
        "meta": meta,
        "rating": rating,
        "runtime": runtime,
        "genres": genres,
        "poster_url": poster_url,
        "poster_path": poster_path or "",
        "release_date": release_date,
        "overview": row.get("overview", "") or "",
        "keywords": keywords,
        "matched": matched,
        "match_count": int(row.get("match_count") or 0),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/languages")
def languages():
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT DISTINCT language FROM movies WHERE language IS NOT NULL ORDER BY language;")
            codes = [row[0] for row in cur.fetchall()]
    except Exception:
        raise HTTPException(status_code=503, detail="languages_unavailable")
    ordered = order_languages(codes)
    return [{"code": code, "name": format_language(code)} for code in ordered]


@app.get("/featured")
def featured(authorization: Optional[str] = Header(None)):
    user = get_optional_user(authorization)
    is_premium = get_user_is_premium(user) if user else False
    cards = [prepare_card(row, []) for row in SAMPLE_FEATURED]
    if not is_premium:
        for card in cards:
            card["overview"] = ""
    return {"featured": cards, "is_premium": is_premium}


@app.post("/search")
def search(payload: SearchRequest, authorization: Optional[str] = Header(None)):
    user = get_optional_user(authorization)
    is_premium = get_user_is_premium(user) if user else False
    results_limit = 50 if is_premium else 10
    raw_query = payload.overview_query or ""
    print(raw_query)
    chosen_tags = map_query_to_tags(raw_query)
    language = payload.language or "Any"
    try:
        rows = fetch_movies(language if language != "Any" else None, chosen_tags, limit=results_limit)
    except Exception:
        raise HTTPException(status_code=503, detail="search_unavailable")
    cards = [prepare_card(row, chosen_tags) for row in rows]
    if not is_premium:
        for card in cards:
            card["overview"] = ""
    return {
        "results": cards,
        "results_count": len(cards),
        "limit": results_limit,
        "matched_tags": chosen_tags,
        "is_premium": is_premium,
    }


@app.post("/auth/login")
def login(payload: AuthRequest):
    if not USER_DATABASE_URL or not JWT_SECRET:
        raise HTTPException(status_code=503, detail="login_unavailable")
    email_clean = normalize_email(payload.email)
    if not email_clean or not payload.password:
        raise HTTPException(status_code=400, detail="missing_credentials")
    user = get_user_by_email(email_clean)
    if not user or not verify_password(payload.password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="invalid_credentials")
    return {"token": create_token(email_clean)}


@app.post("/auth/register")
def register(payload: RegisterRequest):
    if not USER_DATABASE_URL or not JWT_SECRET:
        raise HTTPException(status_code=503, detail="registration_unavailable")
    if not payload.email.strip() or not payload.password:
        raise HTTPException(status_code=400, detail="missing_credentials")
    if payload.password != payload.confirm_password:
        raise HTTPException(status_code=400, detail="passwords_mismatch")
    email_clean = normalize_email(payload.email)
    created = create_user(email_clean, payload.password)
    if not created:
        raise HTTPException(status_code=400, detail="registration_failed")
    return {"token": create_token(email_clean)}


@app.get("/auth/me")
def me(authorization: Optional[str] = Header(None)):
    email = get_required_user(authorization)
    return {"email": email, "is_premium": get_user_is_premium(email)}


@app.get("/favorites")
def list_favorites(authorization: Optional[str] = Header(None)):
    email = get_required_user(authorization)
    if not FAVORITES_DATABASE_URL:
        raise HTTPException(status_code=503, detail="favorites_unavailable")
    rows = get_favorites_for_user(email)
    favorites = []
    for row in rows:
        title = row.get("movie_title", "") or ""
        release_date = str(row.get("release_date") or "")
        poster_path = row.get("poster_path") or ""
        poster_url = f"{IMG_BASE}{poster_path}" if poster_path else ""
        favorites.append(
            {
                "title": title,
                "release_date": release_date,
                "poster_path": poster_path,
                "poster_url": poster_url,
                "genres": row.get("genres") or "",
            }
        )
    return favorites


@app.post("/favorites/toggle")
def toggle_favorites(payload: FavoriteRequest, authorization: Optional[str] = Header(None)):
    email = get_required_user(authorization)
    if not FAVORITES_DATABASE_URL:
        raise HTTPException(status_code=503, detail="favorites_unavailable")

    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="missing_title")

    exists = favorite_exists_for_user(email, title, payload.release_date or "")
    if exists:
        remove_favorite_for_user(email, title, payload.release_date or "")
        return {"favorited": False}

    if not get_user_is_premium(email):
        current_count = count_favorites_for_user(email)
        if current_count >= FREE_FAVORITES_LIMIT:
            raise HTTPException(status_code=403, detail="favorites_limit")

    add_favorite_for_user(
        email,
        title,
        payload.release_date or "",
        payload.poster_path or "",
        payload.genres or "",
    )
    return {"favorited": True}


@app.get("/premium/status")
def premium_status(authorization: Optional[str] = Header(None)):
    email = get_required_user(authorization)
    record = get_premium_record(email)
    is_premium = is_premium_active(record)
    period_end = None
    if record and record.get("current_period_end"):
        try:
            period_end = record["current_period_end"].astimezone(timezone.utc).isoformat()
        except Exception:
            period_end = str(record["current_period_end"])
    return {
        "is_premium": is_premium,
        "cancel_at_period_end": bool(record.get("cancel_at_period_end")) if record else False,
        "current_period_end": period_end,
    }


@app.post("/account/email")
def update_email(payload: UpdateEmailRequest, authorization: Optional[str] = Header(None)):
    email = get_required_user(authorization)
    user = get_user_by_email(email)
    if not user or not verify_password(payload.current_password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="invalid_credentials")
    new_email = normalize_email(payload.new_email)
    updated = update_user_email(email, new_email)
    if not updated:
        raise HTTPException(status_code=400, detail="update_failed")
    update_favorites_email(email, new_email)
    update_premium_email(email, new_email)
    return {"status": "ok"}


@app.post("/account/password")
def update_password(payload: UpdatePasswordRequest, authorization: Optional[str] = Header(None)):
    email = get_required_user(authorization)
    if payload.new_password != payload.confirm_password:
        raise HTTPException(status_code=400, detail="passwords_mismatch")
    user = get_user_by_email(email)
    if not user or not verify_password(payload.current_password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="invalid_credentials")
    updated = update_user_password(email, payload.new_password)
    if not updated:
        raise HTTPException(status_code=400, detail="update_failed")
    return {"status": "ok"}


@app.post("/subscribe")
def subscribe(
    request: Request,
    payload: Optional[SubscribeRequest] = None,
    authorization: Optional[str] = Header(None),
):
    email = get_required_user(authorization)
    if not STRIPE_SECRET_KEY or not TIER_ONE_STRIPE_ID:
        raise HTTPException(status_code=503, detail="stripe_unavailable")
    try:
        import stripe
    except Exception:
        raise HTTPException(status_code=500, detail="stripe_unavailable")

    return_url = ""
    if payload and payload.return_url:
        return_url = payload.return_url.strip()
    if not return_url:
        return_url = STRIPE_RETURN_URL.strip()
    if not return_url:
        base_url = str(request.base_url).rstrip("/")
        return_url = f"{base_url}/"

    stripe.api_key = STRIPE_SECRET_KEY
    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            ui_mode="embedded",
            line_items=[{"price": TIER_ONE_STRIPE_ID, "quantity": 1}],
            return_url=return_url,
            customer_email=email,
            subscription_data={
                "metadata": {"user_email": email},
            },
            metadata={"user_email": email},
        )
    except Exception:
        raise HTTPException(status_code=500, detail="stripe_session_failed")

    client_secret = session.client_secret if session else None
    if not client_secret:
        raise HTTPException(status_code=500, detail="stripe_session_failed")
    return {"clientSecret": client_secret}


@app.post("/subscription/cancel")
def cancel_subscription(authorization: Optional[str] = Header(None)):
    email = get_required_user(authorization)
    record = get_premium_record(email)
    if not is_premium_active(record):
        raise HTTPException(status_code=400, detail="no_active_subscription")

    subscription_id = record.get("stripe_subscription_id") if record else None
    if not subscription_id:
        set_user_premium(email, False, clear_subscription=True)
        return {"status": "canceled"}

    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="stripe_unavailable")
    try:
        import stripe
    except Exception:
        raise HTTPException(status_code=500, detail="stripe_unavailable")

    stripe.api_key = STRIPE_SECRET_KEY
    try:
        subscription = stripe.Subscription.modify(
            subscription_id,
            cancel_at_period_end=True,
        )
    except Exception:
        raise HTTPException(status_code=500, detail="cancel_failed")

    period_end = None
    if subscription and subscription.get("current_period_end"):
        period_end = datetime.fromtimestamp(int(subscription["current_period_end"]), tz=timezone.utc)
    set_user_premium(
        email,
        True,
        subscription_id=subscription_id,
        period_end=period_end,
        cancel_at_period_end=True,
    )
    return {"status": "cancellation_scheduled"}


@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=400, detail="webhook_unconfigured")
    try:
        import stripe
    except Exception:
        raise HTTPException(status_code=500, detail="stripe_unavailable")

    payload = await request.body()
    signature = request.headers.get("stripe-signature", "")
    try:
        event = stripe.Webhook.construct_event(payload, signature, STRIPE_WEBHOOK_SECRET)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid_signature")

    if event.get("type") == "checkout.session.completed":
        session = event.get("data", {}).get("object", {})
        email = session.get("customer_email") or session.get("metadata", {}).get("user_email")
        if email:
            subscription_id = session.get("subscription")
            period_end = None
            cancel_at_period_end = None
            if subscription_id:
                try:
                    subscription = stripe.Subscription.retrieve(subscription_id)
                    if subscription.get("current_period_end"):
                        period_end = datetime.fromtimestamp(
                            int(subscription["current_period_end"]), tz=timezone.utc
                        )
                    cancel_at_period_end = subscription.get("cancel_at_period_end")
                except Exception:
                    subscription = None
            set_user_premium(
                email,
                True,
                subscription_id=subscription_id,
                period_end=period_end,
                cancel_at_period_end=cancel_at_period_end,
            )

    if event.get("type") in {"customer.subscription.updated", "customer.subscription.deleted"}:
        subscription = event.get("data", {}).get("object", {})
        subscription_id = subscription.get("id")
        status = subscription.get("status")
        cancel_at_period_end = subscription.get("cancel_at_period_end")
        period_end = None
        if subscription.get("current_period_end"):
            period_end = datetime.fromtimestamp(int(subscription["current_period_end"]), tz=timezone.utc)

        is_active = status in ACTIVE_SUBSCRIPTION_STATUSES
        if subscription_id:
            if event.get("type") == "customer.subscription.deleted":
                update_premium_by_subscription_id(
                    subscription_id,
                    False,
                    period_end=None,
                    cancel_at_period_end=False,
                    clear_subscription=True,
                )
            else:
                update_premium_by_subscription_id(
                    subscription_id,
                    is_active,
                    period_end=period_end,
                    cancel_at_period_end=cancel_at_period_end,
                )
        else:
            email = subscription.get("metadata", {}).get("user_email")
            if email:
                set_user_premium(
                    email,
                    is_active,
                    subscription_id=subscription_id,
                    period_end=period_end,
                    cancel_at_period_end=cancel_at_period_end,
                    clear_subscription=not is_active,
                )

    return {"status": "ok"}
