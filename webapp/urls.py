from django.urls import path
from . import views

urlpatterns = [
    path("home/", views.Homepage, name="homepage"),
    path("", views.Login, name="login"),
    path("register/", views.Register, name="register"),
    path("logout/", views.Logout, name="logout"),
]
