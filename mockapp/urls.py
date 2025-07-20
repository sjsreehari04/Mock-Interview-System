from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('user/', views.users, name='users'),
    path('experthome/',views.expert_home, name='experthome'),
    path('login/', views.login, name='login'),
    path('register/', views.user, name='user'),
    path('datatable/', views.tableview, name='datatable'),
    path('profile/', views.profile, name='profile'),
    path('edit-profile/', views.editprofile, name='edit_profile'),
    path('expert/',views.expert, name='expert'),
    path('expertprofile/',views.expertprofile,name='expertprofile'),
    path('editexpert/',views.editexpert, name='editexpert'),
    path('viewexpert/',views.view_experts, name='viewexpert'),
    path('tips/',views.tips, name='tips'),
    path('add_tips/', views.add_tips, name='add_tips'),
    path('view_tips/', views.view_tips, name='view_tips'),
    path('tipview/', views.tipview, name='tipview'),
    path('edit_tip/<int:id>/', views.edit_tip, name='edit_tip'),
    path('delete_tip/<int:tip_id>/', views.delete_tip, name='delete_tip'),
    path('chat/', views.chat_page, name='chat'),
    path('get_chat/<int:expert_id>/', views.get_chat, name='get_chat'),
    path('send_message/', views.send_message, name='send_message'),
    path('expert_chat/', views.expert_chat, name='expert_chat'),
    path('get_chat_for_expert/<int:user_id>/', views.get_chat_for_expert, name='get_chat_for_expert'),
    path('send_message_for_expert/', views.send_message_for_expert, name='send_message_for_expert'),
    path('header/', views.header, name='header'),
    path('logout/', views.logout, name='logout'),
    path('analyze_facial_confidence/', views.analyze_facial_confidence, name='analyze_facial_confidence'),
    path('mock/', views.mock, name='mock'),
    path('analyze_speech_tone/', views.analyze_speech_tone, name='analyze_speech_tone'),

    
    path('performance/', views.performance_dashboard, name='performance_dashboard'),
    path('expertresults/<int:user_id>/', views.view_user_results, name='view_user_results'),
    path('view_users_list/', views.view_users_list, name='view_users_list'),
    


   


]







