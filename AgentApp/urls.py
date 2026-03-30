from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing_page, name='landing'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('forgot-password/', views.forgot_password_view, name='forgot_password'),
    # --- Legacy chat endpoint (existing frontend) ---
    path('api/chat/', views.api_chat, name='api_chat'),
    # --- New structured endpoints ---
    path('api/predict/', views.api_predict, name='api_predict'),
    path('api/backtest/', views.api_backtest, name='api_backtest'),
    path('api/models/', views.api_models, name='api_models'),
    # --- Profile & trades ---
    path('save_signal/', views.save_signal, name='save_signal'),
    path('get_profile_data/', views.get_profile_data, name='get_profile_data'),
    path('update_profile_settings/', views.update_profile_settings, name='update_profile_settings'),
    path('profile/', views.profile_view, name='profile'),
    path('get_feedback/', views.get_feedback, name='get_feedback'),
    path('demo/', views.demo, name='demo'),
    path('privacy-policy/', views.privacy_policy_view, name='privacy_policy'),
]
