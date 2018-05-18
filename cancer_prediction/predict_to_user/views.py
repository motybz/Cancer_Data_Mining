from django.shortcuts import render
from .forms import UserEntityForm
from .predict_proba import get_prediction

def index(request):
    form = UserEntityForm()
    return render(request, 'user_input_form.html', {'form': form})

def form_submit(request):
    get_attrs = request.GET
    user_row_data = get_attrs.dict()
    predict_res = get_prediction(user_row_data)
    return render(request, 'predict_res.html', {'predict_res': predict_res})
