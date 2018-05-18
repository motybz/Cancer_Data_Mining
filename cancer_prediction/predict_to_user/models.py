from django.db import models

# Choices:
FEMALE = 'F'
MALE = 'M'
SEX_OPTIONS = (
    (MALE, 'Male'),
    (FEMALE, 'Female'),
)
YES = 'Y'
NO = 'N'
YES_NO_OPTIONS = (
    (YES, 'Yes'),
    (NO, 'No'),
)

class Entity(models.Model):
    sex = models.CharField(max_length=1,choices=SEX_OPTIONS)
    # age = models.IntegerField()
    waist_circumference = models.FloatField()
    # weight = models.FloatField()
    # height = models.FloatField()
    sym_chest_pain = models.IntegerField()
    sym_burning_chest = models.IntegerField()
    sym_sleep_disrupted = models.IntegerField()
    sym_yrs_since_acid_taste_start = models.IntegerField(verbose_name="The number of years since acid taste start")
    sym_taking_stomach_meds = models.CharField(max_length=1,choices=YES_NO_OPTIONS,verbose_name='Are you taking stomach meds?')