from django.urls import path
from . import views

urlpatterns = [
    path("", views.stocks, name="stocks"),
    path("post/new/", views.post_new, name="post_new"),
]
