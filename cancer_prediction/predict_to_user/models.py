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
NEVER = 1
FEW_TIMES_A_YEAR = 2
FEW_TIMES_A_WEEK = 3
DAILY = 4

OCCURENCY_OPTIONS =(
    (NEVER,'Never'),
    (FEW_TIMES_A_YEAR,'Few times a year/month'),
    (FEW_TIMES_A_WEEK,'Few times a week'),
    (DAILY,'Daily')
)
LAST_SIX_MONTHS = 2
SIX_MONTHS_TO_A_YEAR = 3
ONE_TO_TWO_YEARS=4
TWO_TO_FIVE_YEARS=5
FIVE_TO_TEN_YEARS = 6
TEN_TO_TWENTY_YEARS = 7
OVER_TWENTY_YEARS = 8

SINCE_OPTIONS =(
    (NEVER, 'Never'),
    (LAST_SIX_MONTHS,'Last six months'),
    (SIX_MONTHS_TO_A_YEAR,'Six months to a year'),
    (ONE_TO_TWO_YEARS,'One to two years'),
    (TWO_TO_FIVE_YEARS,'Two to five years'),
    (FIVE_TO_TEN_YEARS,'Five to ten years'),
    (TEN_TO_TWENTY_YEARS,'Ten to twenty years'),
    (OVER_TWENTY_YEARS,'Over twenty years'),
)

class Entity(models.Model):
    sym_taking_stomach_meds = models.CharField(max_length=1,choices=YES_NO_OPTIONS,verbose_name='Are you taking stomach meds?')
    sym_yrs_since_acid_taste_start = models.IntegerField(verbose_name="When the acid taste started",choices=SINCE_OPTIONS,blank=True)
    # age = models.IntegerField()
    waist_circumference = models.FloatField(blank=True)
    sex = models.CharField(max_length=1,choices=SEX_OPTIONS,blank=True)
    # weight = models.FloatField()
    sym_chest_pain = models.IntegerField(blank=True,choices=OCCURENCY_OPTIONS)
    sym_burning_chest = models.IntegerField(blank=True,choices=OCCURENCY_OPTIONS)
    height = models.FloatField(blank=True)
    # sym_sleep_disrupted = models.IntegerField(blank=True)

