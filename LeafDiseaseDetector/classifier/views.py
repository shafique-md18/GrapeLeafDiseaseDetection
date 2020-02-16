from django.shortcuts import render
from django.http import HttpResponse, Http404
from django.core.files.storage import FileSystemStorage
from .utils import unique_file_name
from .classify import Predict
from pathlib import Path


def index(request):
    if request.method == 'POST' and request.FILES.get('img', False):
        img = request.FILES['img']
        fs = FileSystemStorage()
        filename = fs.save(unique_file_name(), img)
        uploaded_image_url = fs.url(filename)
        path = Path('.') / 'model_new.h5'
        p = Predict(path)
        u = str(Path('.').parent.parent.absolute()) + uploaded_image_url
        predicted_class = p.predict_image(u)
        context = {
            'uploaded_image_url': uploaded_image_url,
            'predicted_class': predicted_class
        }
        return render(request, 'classifier/results.html', context)
    return render(request, 'classifier/index.html', {})


def handle_uploaded_file(request):
    if request.FILES.get('img', False):
        return render(request, 'classifier/results.html')
    return Http404("Error while processing the image upload.")
