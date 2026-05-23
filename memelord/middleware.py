"""Project middleware."""

from django.contrib.auth.views import redirect_to_login

# Paths that stay reachable without a ratings-app session. /media/ and /thumb/
# are intentionally absent — those views enforce their own @login_required check,
# and sending anonymous users to /login/ is the correct failure mode.
_LOGIN_EXEMPT_PREFIXES = (
    "/login/",
    "/logout/",
    "/static/",
    "/admin/",
)


class LoginRequiredMiddleware:
    """
    Redirect unauthenticated users to LOGIN_URL for every route except login,
    logout, static files, and Django admin.

    Per-view @login_required remains on individual handlers, but this middleware
    is the backstop so a new URL cannot accidentally ship public by omission.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if not request.user.is_authenticated:
            path = request.path
            if not any(path.startswith(prefix) for prefix in _LOGIN_EXEMPT_PREFIXES):
                return redirect_to_login(request.get_full_path())
        return self.get_response(request)
