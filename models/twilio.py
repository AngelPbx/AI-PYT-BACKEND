# schemas/twilio.py
from pydantic import BaseModel
from typing import List

class PhoneNumberOut(BaseModel):
    friendly_name: str

class CountryOut(BaseModel):
    country_code: str
    country: str
