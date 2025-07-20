from django.shortcuts import render,redirect
from .forms import *
from .models import *
from django.contrib import messages
from django.shortcuts import get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from django.utils import timezone
# Create your views here.
def home(request):
    return render(request,'index.html')
def admin(request):
    return render(request,'admin_home.html')
def users(request):
    return render(request,'user_home.html')
def expert_home(request):
    return render(request, 'expert_home.html')

def login(request):
    if request.method == 'POST':
        form =LoginCheck(request.POST)
        print(form)
        if form.is_valid():
            email = form.cleaned_data['email']
            password = form.cleaned_data['password']
            try:
                user = Login.objects.get(email=email) 
                if password==user.password and user.user_type=='user': 
                    request.session['user_id'] = user.id
                    request.session['email'] = user.email
                    return redirect('users') 
                elif user.password == password and user.user_type == 'expert':
                    request.session['expert_id'] = user.id
                    request.session['email'] = user.email
                    return redirect('../experthome/')  
                else:
                    messages.error(request, 'Password is incorrect')
            except Login.DoesNotExist:
                messages.error(request, 'Email is incorrect')
    else:
        form = LoginCheck()
    return render(request, 'login.html', {'form': form})


def user(request):
    if request.method == 'POST':
        user_form = UserForm(request.POST)
        login_form = LoginForm(request.POST)
        if user_form.is_valid() and login_form.is_valid():
            # Save the Login instance first
            login_instance = login_form.save(commit=False)
            login_instance.user_type = 'user'
            login_instance.save()

            # Save the User instance and associate it with the Login instance
            user_instance = user_form.save(commit=False)
            user_instance.Login_id = login_instance  # Correctly assign the Login instance
            user_instance.save()

            messages.success(request, 'User registered successfully!')
            return redirect('home')
    else:
        user_form = UserForm()
        login_form = LoginForm()
    return render(request, 'register.html', {'user_form': user_form, 'login_form': login_form})

def tableview(request):
    users=Users.objects.all()
    # print(users)
    return render(request,'datatable.html',{'users':users})

def profile(request):
    user_id = request.session.get('user_id')
    if not user_id:
        messages.error(request, 'You are not logged in. Please login first.')
        return redirect('login')
    user = Users.objects.get(Login_id=user_id)
    return render(request, 'profile.html', {'user': user})

def editprofile(request):
    user_id = request.session.get('user_id')
    if not user_id:
        messages.error(request, 'User not logged in.')
        return redirect('login')
    
    user = get_object_or_404(Users, Login_id=user_id)
    if request.method == 'POST':
        user_form = UserForm(request.POST, instance=user)
        if user_form.is_valid():
            user_form.save()
            messages.success(request, 'Profile updated successfully!')
            return redirect('profile')
    else:
        user_form = UserForm(instance=user)
    return render(request, 'edit_profile.html', {'user_form': user_form})


def expert(request):
    if request.method == 'POST':
        Expert_form = ExpertForm(request.POST)
        login_form = LoginForm(request.POST)
        if Expert_form.is_valid() and login_form.is_valid():
            # Save the Login instance first
            login_instance = login_form.save(commit=False)
            login_instance.user_type = 'expert'
            login_instance.save()

            # Save the Expert instance and associate it with the Login instance
            expert_instance = Expert_form.save(commit=False)
            expert_instance.Login_id = login_instance  # Assign the Login instance
            expert_instance.save()

            messages.success(request, 'Expert registered successfully!')
            return redirect('home')
    else:
        Expert_form = ExpertForm()
        login_form = LoginForm()
    return render(request, 'expertreg.html', {'user_form': Expert_form, 'login_form': login_form})

def expertprofile(request):
    expert_id = request.session.get('expert_id')
    if not expert_id:
        messages.error(request, 'You are not logged in as an expert. Please login first.')
        return redirect('login')
    user = Experts.objects.get(Login_id=expert_id)
    return render(request, 'expertprofile.html', {'user': user})

def editexpert(request):
    user = Experts.objects.get(Login_id=request.session['expert_id'])
    if request.method == 'POST':
        user_form = ExpertForm(request.POST, instance=user)
        if user_form.is_valid():
            user_form.save()
            messages.success(request, 'Profile updated successfully!')
            return redirect('expertprofile')
    else:
        user_form = ExpertForm(instance=user)
    return render(request, 'edit_expert.html', {'user_form': user_form})

def view_experts(request):
    search_query = request.GET.get('search', '')  # Get the search term from the query parameters
    if search_query:
        experts = Experts.objects.filter(name__icontains=search_query)  # Filter experts by name
    else:
        experts = Experts.objects.all()  # Show all experts if no search term is provided
    return render(request, 'viewexpert.html', {'experts': experts})

def tips(request):
    return render(request, 'tips.html')

def add_tips(request):
    if request.method == 'POST':
        log=request.session['expert_id']
        user=Experts.objects.get(Login_id=log)
        form = InterviewTipForm(request.POST)
        print(form)
        if form.is_valid():
            tip = form.save(commit=False)
            tip.expert = user
            tip.save()
            return redirect('../experthome/')
    else:
        form = InterviewTipForm()
    return render(request, 'tips.html', {'form': form})

def view_tips(request):
    tips = InterviewTips.objects.all().order_by('-date_created')  # Show the latest tips first
    return render(request, 'view_tips.html', {'tips': tips})

from django.shortcuts import render
from .models import InterviewTips

def tipview(request):
    # Ensure the user is logged in as an expert
    expert_id = request.session.get('expert_id')
    if not expert_id:
        return redirect('login')  # Redirect to login if not logged in

    # Retrieve tips created by the logged-in expert
    tips = InterviewTips.objects.filter(expert__Login_id=expert_id).order_by('-date_created')
    return render(request, 'tipview.html', {'tips': tips})

def edit_tip(request, id):
    # Ensure the user is logged in as an expert
    log = request.session.get('expert_id')
    if not log:
        return redirect('login')  # Redirect to login if not logged in

    # Retrieve the tip by its ID and ensure it belongs to the logged-in expert
    tip = get_object_or_404(InterviewTips, id=id, expert__Login_id=log)

    if request.method == 'POST':
        form = InterviewTipForm(request.POST, instance=tip)
        if form.is_valid():
            form.save()
            messages.success(request, 'Tip updated successfully!')
            return redirect('tipview')
    else:
        form = InterviewTipForm(instance=tip)
    return render(request, 'edittip.html', {'form': form})

def delete_tip(request, tip_id):
    # Ensure the user is logged in as an expert
    expert_id = request.session.get('expert_id')
    if not expert_id:
        return redirect('login')  # Redirect to login if not logged in

    # Retrieve the tip by its ID and ensure it belongs to the logged-in expert
    tip = get_object_or_404(InterviewTips, id=tip_id, expert__Login_id=expert_id)

    if request.method == 'POST':
        tip.delete()
        messages.success(request, 'Tip deleted successfully!')
        return redirect('tipview')

    return render(request, 'delete_tip.html', {'tip': tip})

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.db import models
import json

from .models import Chat, Experts, Login, Users

def chat_page(request):
    # Ensure the user is logged in
    user_id = request.session.get('user_id')
    if not user_id:
        return redirect('login')  # Redirect to login if not logged in

    # Fetch all experts
    expert_logins = Login.objects.filter(user_type='expert')
    experts = Experts.objects.filter(Login_id__in=expert_logins)

    return render(request, 'chat.html', {'experts': experts})

def get_chat(request, expert_id):
    login_id = request.session.get('user_id')  # Assuming the user is logged in
    if not login_id:
        return JsonResponse({'error': 'User not logged in'}, status=403)

    messages = Chat.objects.filter(
        (models.Q(sender_id=login_id) & models.Q(receiver_id=expert_id)) |
        (models.Q(sender_id=expert_id) & models.Q(receiver_id=login_id))
    ).order_by('timestamp')

    data = [{
        'sender_id': msg.sender_id,
        'message': msg.message,
        'timestamp': msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')
    } for msg in messages]

    return JsonResponse(data, safe=False)
@csrf_exempt
def send_message(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            sender_id = request.session.get('user_id')  # Get the logged-in user's ID
            receiver_id = data.get('receiver_id')  # Get the receiver ID from the request
            message = data.get('message')  # Get the message content

            if not sender_id:
                return JsonResponse({'status': 'unauthorized'}, status=401)

            # Save the message to the database
            Chat.objects.create(
                sender_id=sender_id,
                receiver_id=receiver_id,
                message=message
            )
            return JsonResponse({'status': 'success'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'failed'}, status=400)

from django.db.models import Q

def expert_chat(request):
    expert_id = request.session.get('expert_id')  # Get the logged-in expert's ID
    if not expert_id:
        return redirect('login')  # Redirect to login if not logged in

    # Fetch unique users who have messaged the expert or received messages from the expert
    user_ids = Chat.objects.filter(
        Q(sender_id=expert_id) | Q(receiver_id=expert_id)
    ).values_list('sender_id', 'receiver_id')

    # Extract unique user IDs excluding the expert's ID
    user_ids = set([uid for pair in user_ids for uid in pair if uid != expert_id])

    # Fetch user details
    users = Users.objects.filter(Login_id__in=user_ids)

    return render(request, 'expert_chat.html', {'users': users, 'expert_id': expert_id})
from django.http import JsonResponse
from django.utils.timezone import localtime

def get_chat_for_expert(request, user_id):
    expert_id = request.session.get('expert_id')  # Get the logged-in expert's ID
    if not expert_id:
        return JsonResponse({'status': 'error', 'message': 'Expert not logged in.'}, status=403)

    # Fetch chat messages between the expert and the user
    chats = Chat.objects.filter(
        sender_id__in=[expert_id, user_id],
        receiver_id__in=[expert_id, user_id]
    ).order_by('timestamp')

    # Serialize chat messages
    chat_data = [
        {
            'sender_id': chat.sender_id,
            'receiver_id': chat.receiver_id,
            'message': chat.message,
            'timestamp': localtime(chat.timestamp).strftime('%Y-%m-%d %H:%M:%S')
        }
        for chat in chats
    ]

    return JsonResponse(chat_data, safe=False)

def send_message_for_expert(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            expert_id = request.session.get('expert_id')  # Get the logged-in expert's ID
            user_id = data.get('user_id')  # Get the user ID from the request
            message = data.get('message')  # Get the message content

            if not expert_id:
                return JsonResponse({'status': 'unauthorized'}, status=401)

            # Save the message to the database
            Chat.objects.create(
                sender_id=expert_id,
                receiver_id=user_id,
                message=message
            )
            return JsonResponse({'status': 'success'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'failed'}, status=400)

def header(request):
    return render(request, 'header.html')

def logout(request):
    # Clear the session data
    request.session.flush()
    messages.success(request, 'You have been logged out successfully.')
    return redirect('login')  # Redirect to the home page after logout

import os
import cv2
import numpy as np
import base64
import tempfile
import ffmpeg
import librosa
import logging
import mediapipe as mp
import speech_recognition as sr
from django.http import JsonResponse
from django.shortcuts import render

logger = logging.getLogger(__name__)

mp_face_mesh = mp.solutions.face_mesh

def mock(request):
    return render(request, 'mockvideo.html')

def analyze_facial_confidence(request):
    if request.method == "POST":
        img_data = request.POST.get('frame')
        _, img_bytes = img_data.split(',')
        img = np.frombuffer(base64.b64decode(img_bytes), np.uint8)
        frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                confidence_score = compute_confidence_from_landmarks(landmarks, frame.shape)
                feedback = generate_confidence_feedback(confidence_score)

                return JsonResponse({
                    'confidence_score': round(confidence_score, 2),
                    'feedback': feedback
                })

        return JsonResponse({'error': 'No face detected'})

def compute_confidence_from_landmarks(landmarks, image_shape):
    h, w, _ = image_shape

    top_lip = landmarks[13]
    bottom_lip = landmarks[14]
    mouth_openness = np.abs((bottom_lip.y - top_lip.y) * h)

    left_brow = landmarks[65]
    left_eye = landmarks[159]
    eyebrow_raise = np.abs((left_brow.y - left_eye.y) * h)

    left_upper_eyelid = landmarks[159]
    left_lower_eyelid = landmarks[145]
    eye_openness = np.abs((left_lower_eyelid.y - left_upper_eyelid.y) * h)

    left_inner_brow = landmarks[55]
    right_inner_brow = landmarks[285]
    brow_distance = np.abs((right_inner_brow.x - left_inner_brow.x) * w)

    confidence = 100
    confidence -= mouth_openness * 5
    confidence -= eyebrow_raise * 2

    if brow_distance < w * 0.06:
        confidence -= 20

    if eye_openness > h * 0.04:
        confidence -= 15

    return max(0, min(100, confidence))


def generate_confidence_feedback(score):
    if score < 50:
        return (
            "Your facial cues suggest low confidence. "
            "Relax your jaw, keep your mouth closed when not speaking, "
            "and maintain steady eye contact."
        )
    return "Facial expressions indicate confidence. Great job!"

def convert_audio_to_wav(input_path):
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    output_path = tmp_wav.name
    tmp_wav.close()
    try:
        (
            ffmpeg
            .input(input_path)
            .output(output_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
            .overwrite_output()
            .run(quiet=True)
        )
        return output_path
    except ffmpeg.Error as e:
        raise RuntimeError(f"Audio conversion failed: {e}")
    
from .models import MockScore


def analyze_speech_tone(request):
    if request.method != "POST":
        return JsonResponse({'error': 'Invalid request method'}, status=405)

    userid = request.session.get('user_id')
    if not userid:
        logger.error("No user_id found in session.")
        return JsonResponse({'error': 'User not authenticated. Please log in again.'}, status=403)

    uid = get_object_or_404(Login, id=userid)

    category = request.POST.get('question_category')
    question_text = request.POST.get('question_text')
    facial_confidence = request.POST.get('facial_confidence')

    logger.info(f"Speech analysis POST: user={uid}, category={category}, question={question_text}, facial_confidence={facial_confidence}")

    if not category or not question_text or facial_confidence is None:
        return JsonResponse({'error': 'Category, question, and facial confidence are required.'}, status=400)

    try:
        facial_confidence = float(facial_confidence)
    except ValueError:
        return JsonResponse({'error': 'Invalid facial confidence value.'}, status=400)

    if 'audio' not in request.FILES:
        logger.error("No audio file uploaded")
        return JsonResponse({'error': 'No audio file uploaded'})

    audio_file = request.FILES['audio']
    temp_webm_path, temp_wav_path = None, None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            for chunk in audio_file.chunks():
                tmp.write(chunk)
            temp_webm_path = tmp.name
        logger.info(f"Saved uploaded audio to {temp_webm_path}")

        temp_wav_path = convert_audio_to_wav(temp_webm_path)
        logger.info(f"Converted audio to WAV at {temp_wav_path}")

        y, sr_rate = librosa.load(temp_wav_path, sr=None)
        duration_seconds = librosa.get_duration(y=y, sr=sr_rate)
        logger.info(f"Recording duration: {duration_seconds:.2f} seconds")

        if duration_seconds < 1.5:
            return JsonResponse({'error': 'Recording too short. Please speak at least 2 seconds.'})

        avg_energy = np.mean(np.abs(y))
        if avg_energy < 0.001:
            return JsonResponse({'error': 'No speech detected. Please speak clearly.'})

        pitches, magnitudes = librosa.piptrack(y=y, sr=sr_rate)
        pitch_values = pitches[pitches > 0]
        avg_pitch, pitch_std = (
            (float(np.mean(pitch_values)), float(np.std(pitch_values)))
            if len(pitch_values) > 0 else (0.0, 0.0)
        )
        logger.info(f"Average pitch: {avg_pitch:.1f} Hz, Pitch std dev: {pitch_std:.1f}")

        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_wav_path) as source:
            audio_data = recognizer.record(source)
            try:
                transcript = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                transcript = ""

        word_count = len(transcript.split())
        if word_count < 1:
            return JsonResponse({'error': 'No recognizable speech detected. Please speak clearly.'})

        speech_rate_wpm = float(word_count / (duration_seconds / 60)) if duration_seconds > 0 else 0.0
        logger.info(f"Speech rate: {speech_rate_wpm:.1f} WPM")

        speech_confidence = max(0, min(100, 100 - abs(140 - speech_rate_wpm)))  # ideal ~140 WPM
        combined_confidence = round((speech_confidence + facial_confidence) / 2, 1)

        feedback = generate_speech_feedback(speech_rate_wpm, avg_pitch, pitch_std)

        MockScore.objects.create(
            login=uid,
            question_category=category,
            question_text=question_text,
            speech_confidence=speech_confidence,
            facial_confidence=facial_confidence,
            combined_confidence=combined_confidence,
        )

        return JsonResponse({
            'transcript': transcript,
            'speech_rate_wpm': round(speech_rate_wpm, 1),
            'avg_pitch_hz': round(avg_pitch, 1),
            'speech_confidence': round(speech_confidence, 1),
            'combined_confidence': combined_confidence,
            'feedback': feedback,
        })

    except RuntimeError as e:
        logger.exception("Speech analysis error")
        return JsonResponse({'error': str(e)})
    except Exception as e:
        logger.exception("Speech analysis error")
        return JsonResponse({'error': f'Error analyzing speech: {str(e)}'})
    finally:
        if temp_webm_path and os.path.exists(temp_webm_path):
            os.unlink(temp_webm_path)
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)

#
def generate_speech_feedback(speech_rate_wpm, avg_pitch, pitch_std=None):
    tips = []

    # ✅ Speech rate feedback
    if speech_rate_wpm < 100:
        tips.append(
            "Your speech pace is a bit slow. Consider increasing your speed slightly (aim for 120–160 WPM) to sound more confident and engaged."
        )
    elif speech_rate_wpm > 180:
        tips.append(
            "You are speaking very quickly. Slowing down a bit will make your message clearer and easier to follow."
        )
    else:
        tips.append("Your speech rate is within a confident and natural range. Great job!")

    # ✅ Pitch feedback with nuance
    if avg_pitch > 300:
        if pitch_std and pitch_std > 50:  # unstable high pitch
            tips.append(
                "Your tone is quite high and varies significantly, which can sometimes come across as nervousness. A steadier, lower tone can project calmness."
            )
        else:
            tips.append(
                "Your tone is naturally high but steady — that’s absolutely fine as long as you feel comfortable."
            )
    elif avg_pitch < 100:
        tips.append(
            "Your tone sounds quite flat. Introducing more variation in your intonation can make you sound more confident and engaging."
        )
    else:
        tips.append("Your tone has a pleasant pitch and stability, conveying confidence effectively.")

    return " ".join(tips)

from django.db.models import Avg
from django.shortcuts import render, get_object_or_404
from .models import MockScore, Login

from django.shortcuts import render, get_object_or_404
from django.db.models import Avg
from .models import MockScore, Login

def performance_dashboard(request):
    user_id = request.session.get('user_id')
    login = get_object_or_404(Login, id=user_id)

    # Fetch individual scores for the logged-in user
    scores = MockScore.objects.filter(login=login).order_by('created_at')

    # Aggregate average scores by question_category
    summary = (
        scores
        .values('question_category')
        .annotate(
            avg_speech=Avg('speech_confidence'),
            avg_facial=Avg('facial_confidence'),
            avg_combined=Avg('combined_confidence')
        )
        .order_by('question_category')
    )

    # Add verdict to each category summary item
    for item in summary:
        avg = item['avg_combined']
        if avg >= 75:
            verdict = 'Excellent'
        elif avg >= 50:
            verdict = 'Good Work'
        else:
            verdict = 'Needs Improvement'
        item['verdict'] = verdict

    # Pass both individual scores and the category summary to the template
    return render(
        request,
        'performance_dashboard.html',
        {'scores': scores, 'summary': summary}
    )

from django.contrib.auth.decorators import login_required

def view_users_list(request):
    expert_id = request.session.get('expert_id')
    if not expert_id:
        messages.error(request, "You must be logged in as an expert to view this page.")
        return redirect('login')

    users = Users.objects.all().select_related('Login_id')  # fetch user info with login
    return render(request, 'expert_users.html', {'users': users})  # updated file name


def view_user_results(request, user_id):
    expert_id = request.session.get('expert_id')
    if not expert_id:
        messages.error(request, "You must be logged in as an expert to view this page.")
        return redirect('login')

    login = get_object_or_404(Login, id=user_id)
    scores = MockScore.objects.filter(login=login).order_by('-created_at')

    return render(request, 'expert_user_results.html', {'scores': scores, 'user': login})  # already correct

# def view_users_list(request):
#     expert_id = request.session.get('expert_id')
#     if not expert_id:
#         messages.error(request, "You must be logged in as an expert to view this page.")
#         return redirect('login')

#     users = Users.objects.all().select_related('Login_id')
#     return render(request, 'expert_users.html', {'users': users})

# def view_user_results(request, user_id):
#     expert_id = request.session.get('expert_id')
#     if not expert_id:
#         messages.error(request, "You must be logged in as an expert to view this page.")
#         return redirect('login')

#     login = get_object_or_404(Login, id=user_id)
#     scores = MockScore.objects.filter(login=login).order_by('-created_at')

#     return render(request, 'expert_user_results.html', {'scores': scores, 'user': login})
