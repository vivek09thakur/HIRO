from django.contrib.auth.models import AbstractUser, Group, Permission
from django.db import models
from django.utils import timezone

class CustomUser(AbstractUser):
    is_online = models.BooleanField(default=False)
    last_login = models.DateTimeField(default=timezone.now)
    groups = models.ManyToManyField(
        Group,
        related_name="custom_user_groups", 
        blank=True
    )
    user_permissions = models.ManyToManyField(
        Permission,
        related_name="custom_user_permissions", 
    )

    def get_absolute_url(self):
        return reverse('chat-app')  # Redirect to chat app after login

class ChatHistory(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    message = models.TextField()
    response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']