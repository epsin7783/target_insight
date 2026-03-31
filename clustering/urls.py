from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('analyze/', views.analyze_view, name='analyze'),
    path('result/<int:session_id>/', views.session_result_view, name='session_result'),
    path('sample-csv/', views.sample_csv_view, name='sample_csv'),
]
