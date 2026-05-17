import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


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

DEBUG = os.environ.get("DJANGO_DEBUG", "true").lower() != "false"

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
    "timeout": 3600,
    "retry": 25200,
    "max_attempts": 1,
    "orm": "default",
}

MEDIA_ROOT = DATA_DIR
MEDIA_URL = "/media/"

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [BASE_DIR / "static"]

WEIGHTS_PATH = DATA_DIR / "Janulon_weights.pkl"
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", BASE_DIR / "config.toml"))

LOGIN_URL = "/login/"
LOGIN_REDIRECT_URL = "/rate/inbox/"
LOGOUT_REDIRECT_URL = "/login/"

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
