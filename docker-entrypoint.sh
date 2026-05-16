#!/bin/sh
set -e

case "$1" in
  web)
    python manage.py migrate --run-syncdb
    python manage.py collectstatic --noinput --clear -v 0
    exec gunicorn memelord.wsgi:application \
      --bind 0.0.0.0:8000 \
      --workers 2 \
      --timeout 300 \
      --access-logfile -
    ;;
  *)
    exec python manage.py "$@"
    ;;
esac
