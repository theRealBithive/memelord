import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "django-insecure--wk7gr=8t^oipj2-dnemg0@pc1(p&=p7q2)ho@54+*x&$(9mu4",
)

DEBUG = os.environ.get("DJANGO_DEBUG", "true").lower() != "false"

ALLOWED_HOSTS = os.environ.get("ALLOWED_HOSTS", "localhost,127.0.0.1,testserver").split(",")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django_htmx",
    "ratings",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
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
    }
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
