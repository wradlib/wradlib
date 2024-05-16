from django import forms

from .models import Location

class MyForm(forms.ModelForm):
    class Meta:
        model = Location
        fields = ("name", "latitude", "longitude", "height_tower", "vertical_angle")
