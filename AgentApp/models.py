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

    def __str__(self):
        return f"{self.user.username} - {self.symbol} ({self.signal})"

class TraderProfile(models.Model):
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE, related_name='profile')
    
    # Performance Metrics
    total_profit_loss = models.FloatField(default=0.0)
    win_rate = models.FloatField(default=0.0) # percentage
    risk_reward_ratio = models.FloatField(default=0.0)
    max_drawdown = models.FloatField(default=0.0)
    sharpe_ratio = models.FloatField(default=0.0)
    trade_accuracy = models.FloatField(default=0.0) # directional accuracy
    
    # Automation & Execution Controls
    auto_trading = models.BooleanField(default=False)
    
    MODE_CHOICES = [
        ('MANUAL', 'Manual'),
        ('SEMI', 'Semi-auto'),
        ('FULL', 'Full auto'),
    ]
    mode = models.CharField(max_length=10, choices=MODE_CHOICES, default='MANUAL')
    
    allowed_symbols = models.TextField(default="EUR/USD,GBP/USD,USD/JPY,EUR/JPY")
    allowed_timeframes = models.TextField(default="15min,30min,1h")
    
    STRATEGY_CHOICES = [
        ('FIXED', 'Fixed lot'),
        ('RISK', 'Risk-based'),
    ]
    trade_size_strategy = models.CharField(max_length=10, choices=STRATEGY_CHOICES, default='FIXED')
    fixed_lot_size = models.FloatField(default=0.01)
    risk_per_trade = models.FloatField(default=1.0) # percentage
    
    def __str__(self):
        return f"Profile of {self.user.username}"




