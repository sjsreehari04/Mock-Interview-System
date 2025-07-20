from django.db import models
from django.utils.timezone import now
from django.utils import timezone
from django.contrib.auth.models import User



# Create your models here.
class Login(models.Model):
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=100)
    user_type=models.CharField(max_length=10)
    status=models.BooleanField(default=0)
class Users(models.Model):
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]

    name = models.CharField(max_length=100)
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    dob = models.DateField()
    contact = models.CharField(max_length=15)
    Login_id = models.OneToOneField(Login, on_delete=models.CASCADE)

class Experts(models.Model):
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]

    name = models.CharField(max_length=100)
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    dob = models.DateField()
    contact = models.CharField(max_length=15)
    field=models.CharField(max_length=100,default=0)
    experience=models.CharField(max_length=100,default=0)
    Login_id = models.OneToOneField(Login, on_delete=models.CASCADE)


class InterviewTips(models.Model):
    expert = models.ForeignKey(Experts, on_delete=models.CASCADE)  # Link to the expert's login
    tip = models.TextField()  # The interview tip
    date_created = models.DateTimeField(default=now)  # Automatically store the creation date

    def __str__(self):
        return f"Tip by {self.expert.email} on {self.date_created}"

class Chat(models.Model):
    sender_id = models.IntegerField()
    receiver_id = models.IntegerField()
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"From {self.sender_id} to {self.receiver_id}: {self.message}"
class MockScore(models.Model):
    login = models.ForeignKey(Login, on_delete=models.CASCADE)
    question_category = models.CharField(max_length=100)
    question_text = models.TextField()
    speech_confidence = models.FloatField()
    facial_confidence = models.FloatField()
    combined_confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.login.username} - {self.question_category} ({self.combined_confidence:.1f}%)"

# class ChatMessage(models.Model):
#     user = models.ForeignKey(Users, on_delete=models.CASCADE)
#     expert = models.ForeignKey(Experts, on_delete=models.CASCADE)
#     sender = models.CharField(max_length=10, choices=[('user', 'User'), ('expert', 'Expert')])
#     message = models.TextField()
#     timestamp = models.DateTimeField(default=timezone.now)

#     def __str__(self):
#         return f"{self.sender}: {self.message[:30]}"