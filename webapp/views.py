from django.shortcuts import render
from HIRO.HIRO import HEALTHCARE_COMPANION

# Create your views here.
def Homepage(request):
    return render(request, 'webapp/index.html')