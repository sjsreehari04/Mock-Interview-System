from django import forms
from .models import *

class LoginForm(forms.ModelForm):
    class Meta:
        model = Login
        fields = ['email', 'password']
        widgets = {
            'password': forms.PasswordInput(attrs={'placeholder': 'Enter your password'}),
        }

class UserForm(forms.ModelForm):
    class Meta:
        model = Users
        fields = ['name', 'gender', 'dob', 'contact']  # Corrected field name
        widgets = {
            'dob': forms.DateInput(attrs={'type': 'date'}),
        }
class LoginCheck(forms.Form):
    email = forms.EmailField()
    password = forms.CharField(widget=forms.PasswordInput)
class ExpertForm(forms.ModelForm):
    field_CHOICES=[
            ('IT', 'IT'),
            ('HR', 'HR'),
            ('Finance', 'Finance'),
            ('Marketing', 'Marketing'),
            ('Other', 'Other')
        ]
    field=forms.ChoiceField(choices=field_CHOICES)
    class Meta:
        model = Experts
        
        fields = ['name', 'gender', 'dob', 'contact', 'field', 'experience']
        widgets = {
            'dob': forms.DateInput(attrs={'type': 'date'}),
        }

class InterviewTipForm(forms.ModelForm):
    class Meta:
        model = InterviewTips
        fields = ['tip']
        widgets = {
            'tip': forms.Textarea(attrs={'placeholder': 'Write your interview tips here...', 'rows': 5}),
        }

class ChatForm(forms.ModelForm):
    class Meta:
        model = Chat
        fields = ['message']
        widgets = {
            'message': forms.Textarea(attrs={'placeholder': 'Type your message here...', 'rows': 3}),
        }