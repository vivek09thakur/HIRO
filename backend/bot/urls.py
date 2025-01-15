from django.urls import path
from .views import ChatHistoryView

urlpatterns = [
    path('chat/', ChatHistoryView.as_view()),
]