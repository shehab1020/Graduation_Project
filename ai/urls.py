from django.urls import path
from . import views


app_name = 'ai'


urlpatterns = [
    path('questions/', views.get_questions),
    path('evaluate/', views.evaluate),
    path('roadmap/', views.generate_roadmap),
    path('options/', views.get_options),
]