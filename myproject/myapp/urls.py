from django.urls import path
from .views import UploadFileView, ProcessCSVView, ChatView, AnswerView

urlpatterns = [
    path('upload_file/', UploadFileView.as_view(), name='upload_file'),
    path('process_csv/', ProcessCSVView.as_view(), name='process_csv'),
    path('chat/', ChatView.as_view(), name='chat'),
    path('url_chat/', AnswerView.as_view(), name='url_chat'),
]