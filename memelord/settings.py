import os
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _resolve_app_version() -> str:
    """
    Version string shown in the UI footer / nav menu.

    Precedence: APP_VERSION env var (set by the Docker build from the git tag in
    CI) → `git describe` against the working tree (so developers running
    `make run` see e.g. v1.10.3-2-gabc123-dirty without rebuilding) → "dev".
    The git fallback is wrapped because Docker images strip the .git directory
    via .dockerignore, and we don't want a crash there.
    """
    env = os.environ.get("APP_VERSION", "").strip()
    if env:
        return env
    try:
        out = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        described = out.stdout.strip()
        if described:
            return described
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass
    return "dev"


APP_VERSION = _resolve_app_version()


def _csv_env(name: str, default: str = "") -> list[str]:
    """Parse a comma-separated environment variable into a stripped list."""
    return [
        part.strip()
        for part in os.environ.get(name, default).split(",")
        if part.strip()
    ]


def _https_csrf_origins_for_hosts(hosts: list[str]) -> list[str]:
    """Build https:// CSRF trusted origins for public hostnames (not local dev)."""
    local = {"localhost", "127.0.0.1", "testserver", "[::1]"}
    return [
        f"https://{host}"
        for host in hosts
        if host not in local and not host.startswith(".")
    ]


SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "django-insecure--wk7gr=8t^oipj2-dnemg0@pc1(p&=p7q2)ho@54+*x&$(9mu4",
)

DEBUG = os.environ.get("DJANGO_DEBUG", "false").lower() == "true"

if not DEBUG:
    _placeholder_prefixes = ("django-insecure-", "change-me")
    if any(SECRET_KEY.startswith(p) for p in _placeholder_prefixes):
        from django.core.exceptions import ImproperlyConfigured

        raise ImproperlyConfigured(
            "DJANGO_SECRET_KEY must be set to a real secret before running in production. "
            'Generate one with: python -c "import secrets; print(secrets.token_urlsafe(50))"'
        )

ALLOWED_HOSTS = _csv_env("ALLOWED_HOSTS", "localhost,127.0.0.1,testserver")

CSRF_TRUSTED_ORIGINS = _csv_env("CSRF_TRUSTED_ORIGINS")
if not CSRF_TRUSTED_ORIGINS and not DEBUG:
    CSRF_TRUSTED_ORIGINS = _https_csrf_origins_for_hosts(ALLOWED_HOSTS)

if not DEBUG:
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django_htmx",
    "django_q",
    "ratings",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "memelord.middleware.LoginRequiredMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "django_htmx.middleware.HtmxMiddleware",
]

ROOT_URLCONF = "memelord.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.template.context_processors.media",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "ratings.context_processors.notification_config",
                "ratings.context_processors.app_version",
            ],
        },
    },
]

WSGI_APPLICATION = "memelord.wsgi.application"

DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": DATA_DIR / "memelord.db",
        "OPTIONS": {"timeout": 20},
    }
}

Q_CLUSTER = {
    "name": "memelord",
    "workers": 1,
    "timeout": 14400,
    "retry": 28800,
    "max_attempts": 1,
    "orm": "default",
}

MEDIA_ROOT = DATA_DIR
MEDIA_URL = "/media/"

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [BASE_DIR / "static"]

WEIGHTS_PATH = DATA_DIR / "Janulon_weights.pkl"
NSFW_WEIGHTS_PATH = DATA_DIR / "Janulon_nsfw_weights.pkl"
PHASH_MAX_DISTANCE = int(os.environ.get("PHASH_MAX_DISTANCE", "5"))
DINO_DEDUP_COSINE_THRESHOLD = float(
    os.environ.get("DINO_DEDUP_COSINE_THRESHOLD", "0.92")
)
NSFW_THRESHOLD = float(os.environ.get("NSFW_THRESHOLD", "0.30"))
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", BASE_DIR / "config.toml"))


def _load_vision_thresholds() -> tuple[int, int]:
    """Read 1-6 SFW/NSFW review-queue thresholds from config.toml's [vision] section.

    Loaded here (not lazily) so views and counters get a stable settings value
    without re-parsing TOML per request. Missing file or section both default
    to bucket 1 (show all), so an upgraded deployment without an updated
    config.toml behaves exactly as before.
    """
    import tomllib

    try:
        with open(CONFIG_PATH, "rb") as f:
            cfg = tomllib.load(f)
    except (FileNotFoundError, OSError):
        return 1, 1
    vision = cfg.get("vision", {})

    def _clamp(value) -> int:
        try:
            return max(1, min(6, int(value)))
        except (TypeError, ValueError):
            return 1

    return _clamp(vision.get("sfw_threshold", 1)), _clamp(
        vision.get("nsfw_threshold", 1)
    )


SFW_THRESHOLD_BUCKET, NSFW_THRESHOLD_BUCKET = _load_vision_thresholds()

LOGIN_URL = "/login/"
LOGIN_REDIRECT_URL = "/review/"
LOGOUT_REDIRECT_URL = "/login/"

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
