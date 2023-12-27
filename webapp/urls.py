from django.urls import path
from . import views

urlpatterns = [
    path("", views.Homepage, name="homepage"),
    path("login/", views.Login, name="login"),
    path("register/", views.Register, name="register"),
]
