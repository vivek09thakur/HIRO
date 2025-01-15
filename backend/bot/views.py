from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import ChatHistory
from .serializers import ChatHistorySerializer

class ChatHistoryView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        chats = ChatHistory.objects.filter(user=request.user).order_by('-timestamp')
        serializer = ChatHistorySerializer(chats, many=True)
        return Response(serializer.data)

    def post(self, request):
        data = request.data
        chat = ChatHistory.objects.create(
            user=request.user,
            message=data['message'],
            response="This is a response",  # Replace with chatbot logic
        )
        serializer = ChatHistorySerializer(chat)
        return Response(serializer.data)
