
from django.shortcuts import redirect
from django.urls import reverse

class LoginRequiredMiddleware:
    """
    Middleware that requires a user to be logged in to access certain pages.
    """
    def init(self, get_response):
        self.get_response = get_response
        self.exempt_urls = [
            reverse('login'),  # Add names of views that should be exempt from login requirement
            reverse('register'),
        ]

    def call(self, request):
        if not request.session.get('user_id') and request.path not in self.exempt_urls:
            return redirect('login')  # Replace 'login' with the name of your login view

        return self.get_response(request)