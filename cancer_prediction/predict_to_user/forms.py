from django.forms import ModelForm
from .models import Entity
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit, Layout,Button
from crispy_forms.bootstrap import *



class UserEntityForm(ModelForm):
    helper = FormHelper()
    helper.form_class = 'form-horizontal'
    helper.form_id = 'user_form'
    helper.label_class = 'col-lg-2'
    helper.field_class = 'col-lg-8'
    helper.form_method = 'GET'
    helper.add_input(Submit('submit', 'Submit'))
    class Meta:
        model = Entity
        exclude = []



