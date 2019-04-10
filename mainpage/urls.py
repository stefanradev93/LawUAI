from django.urls import path
from . import views
from django.conf.urls import url

urlpatterns = [
    path('', views.home, name="home-page"),
    url(r'^classify/$', views.classify, name='classify')

]
