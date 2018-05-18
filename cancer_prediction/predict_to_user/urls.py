from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('form_submit/', views.form_submit, name='predict_results'),
]