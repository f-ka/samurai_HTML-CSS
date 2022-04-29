from django.shortcuts import render, redirect
from .forms import PostForm
from django.utils import timezone
from sklearn.externals import joblib
from django.http import HttpResponse


def post_list(request):
    return render(request, "app/post_list.html", {})


def post_new(request):
    form = PostForm()
    return render(request, "app/post_edit.html", {"form": form})
