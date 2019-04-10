from django.shortcuts import render
from django.http import HttpResponse
try:
    from django.utils import simplejson as json
except ImportError:
    import json
import os
import  numpy as np
from .genrator import TitleGenerator
from .classifier import AttentionAoLClassifier
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import tensorflow as tf
tf.enable_eager_execution()

title_generator = TitleGenerator()
classifier= AttentionAoLClassifier()
categories = ['Arbeitsrecht', 'Erbrecht', 'Familienrecht', 'Kaufrecht', 'Mietrecht', 'Öffentliches Recht', 'Sozialversicherungsrecht', 'Steuerrecht', 'Strafrecht', 'Vertragsrecht']
#categories=['Arbeitsrecht', 'Kaufrecht', 'Steuerrecht', 'Familienrecht', 'Oeffentlichesrecht', 'Strafrecht', 'Erbrecht', 'Mietrecht _ Wohnungseigentum', 'Sozialversicherungsrecht', 'Vertragsrecht']

def home(request):
    return render(request, 'mainpage/index2.html')

def rescale_linear(array, new_min, new_max):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b

def classify(request):
    if request.method == 'GET':
        text = request.GET.get('text', None)
        t=title_generator.generate_title(text)
        print(t)
        probs, attn_weights,decoded=classifier.classify_text(text)
        probs=probs*100
        attn_weights=attn_weights.squeeze()[0:len(decoded)]
        attn_weights=255-rescale_linear(attn_weights,0,255)

        decoded_with_tages=[]
        for d in decoded:
            decoded_with_tages.append("<label style='background-color:rgb(219, 48, 73)'>"+d+"</label>")
    return HttpResponse(json.dumps({"classes": probs.tolist()[0],"class":categories[np.argmax(probs)], "title": t,"question":' '.join(decoded_with_tages),"classifier_attn":attn_weights.tolist()}),
                        content_type='application/json')


