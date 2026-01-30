from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from .forms import SignupForm, LoginForm
import json
import os
import sys
from .inference import ModelInference
from .llm_service import LLMService
from decouple import config
from django.conf import settings
from .models import Trade
def signup_view(request):
    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('dashboard')
    else:
        form = SignupForm()
    return render(request, 'signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = LoginForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('dashboard')
    else:
        form = LoginForm()
    return render(request, 'login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('login')

def forgot_password_view(request):
    
    return render(request, 'forgot_password.html')

@login_required
def dashboard_view(request):
    return render(request, 'index.html', {'user': request.user})

def landing_page(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    return render(request, 'landing.html')

def demo(request):
    return render(request,'index.html')


@csrf_exempt
def api_chat(request):
    print("API Chat Request Received")
    sys.stdout.flush()
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_prompt = data.get('prompt', '')
            # Use provided key or fallback to server-side key
            api_key = data.get('api_key') or settings.GROQ_API_KEY
            
            if not api_key:
                return JsonResponse({"error": "Groq API Key is required"}, status=400)
                
            # Initialize services
            llm_service = LLMService(api_key=api_key)
            inference_service = ModelInference(api_key=settings.TWELVE_DATA_API_KEY)
            
            # 1. Parse Intent
            intent = llm_service.parse_intent(user_prompt)
            symbol = intent.get("symbol", "EUR/USD")
            timeframe = intent.get("timeframe", "30min")
            
            # 2. Run Inference
            try:
                prediction_data = inference_service.predict(symbol, timeframe)
            except Exception as e:
                print(f"Inference Error: {str(e)}")
                sys.stdout.flush()
                return JsonResponse({"error": f"Inference Error: {str(e)}"}, status=500)
            
            # 3. Generate Response
            response_text = llm_service.generate_response(user_prompt, prediction_data)
            
            return JsonResponse({
                "response": response_text,
                "data": prediction_data
            })
            
        except Exception as e:
            print(f"API Error: {str(e)}")
            sys.stdout.flush()
            return JsonResponse({"error": str(e)}, status=500)
    
    return JsonResponse({"error": "Invalid method"}, status=405)


@csrf_exempt
def save_signal(request):
    if request.method == "POST":
        data = json.loads(request.body)

        Trade.objects.create(
            signal=data["signal"],
            pair=data["pair"],
            timeframe=data["timeframe"],
            price=data["price"],
            action=data["action"],
            tp=data["tp"],
            sl=data["sl"]
        )

        return JsonResponse({"status": "success"})
    

def get_feedback(request):
    return render(request,"feedback.html")