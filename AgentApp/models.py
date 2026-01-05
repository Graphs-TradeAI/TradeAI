from django.contrib.auth.models import AbstractUser
from django.db import models
from django.contrib import admin

class CustomUser(AbstractUser):
    date_of_birth = models.DateField(null=True, blank=True)

    def __str__(self):
        return self.username

class Trade(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    symbol = models.CharField(max_length=10)
    timeframe = models.CharField(max_length=10)
    price = models.FloatField()
    target = models.FloatField()
    tp = models.FloatField()
    sl = models.FloatField()
    signal = models.CharField(max_length=10)
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)




