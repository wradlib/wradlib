from django.db import models

# Create your models here.


class Location(models.Model):
    name = models.CharField(max_length=100)
    latitude = models.DecimalField(max_digits=40, decimal_places=18)
    longitude = models.DecimalField(max_digits=40, decimal_places=18)
    height_tower = models.IntegerField(blank=True, null=True)
    vertical_angle = models.DecimalField(max_digits=5, decimal_places=2)
    image_beam = models.ImageField(default="default.jpg", upload_to="senarios")
    image_couverture = models.ImageField(default="default.jpg", upload_to="senarios")

    def __str__(self):
        return " Radar de " + self.name
