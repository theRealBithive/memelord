from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import include, path

from memelord.media import serve_media

urlpatterns = [
    path("admin/", admin.site.urls),
    path(
        "login/", auth_views.LoginView.as_view(template_name="login.html"), name="login"
    ),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("media/<path:file_path>", serve_media, name="serve_media"),
    path("", include("ratings.urls")),
]
