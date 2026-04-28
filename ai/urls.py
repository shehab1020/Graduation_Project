from django.urls import path
from . import views


app_name = 'ai'


urlpatterns = [
    path('questions/', views.get_questions),
    path('evaluate/', views.evaluate),
    # path('assessment/', views.submit_assessment),
]