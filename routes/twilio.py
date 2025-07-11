from fastapi import APIRouter, Query
from typing import List, Optional
from utils.twilio_client import twilio_client
from models.twilio import PhoneNumberOut, CountryOut

router = APIRouter()

@router.get("/twilio/countries", response_model=List[CountryOut])
def get_available_countries():
    countries = twilio_client.available_phone_numbers.list(limit=100)
    return [{"country_code": c.country_code, "country": c.friendly_name} for c in countries]


@router.get("/twilio/local-numbers", response_model=List[PhoneNumberOut])
def get_local_numbers(
    country: str = Query("US"),
    area_code: Optional[int] = None,
    contains: Optional[str] = None,
    limit: int = Query(20, le=100),
):
    filters = {"limit": limit}
    if area_code:
        filters["area_code"] = area_code
    if contains:
        filters["contains"] = contains

    numbers = twilio_client.available_phone_numbers(country).local.list(**filters)
    return [{"friendly_name": n.friendly_name} for n in numbers]


@router.get("/twilio/tollfree-numbers", response_model=List[PhoneNumberOut])
def get_toll_free_numbers(
    country: str = Query("US"),
    contains: Optional[str] = None,
    limit: int = Query(20, le=100),
):
    filters = {"limit": limit}
    if contains:
        filters["contains"] = contains

    numbers = twilio_client.available_phone_numbers(country).toll_free.list(**filters)
    return [{"friendly_name": n.friendly_name} for n in numbers]


@router.get("/twilio/mobile-numbers", response_model=List[PhoneNumberOut])
def get_mobile_numbers(
    country: str = Query("GB"),
    limit: int = Query(20, le=100),
):
    numbers = twilio_client.available_phone_numbers(country).mobile.list(limit=limit)
    return [{"friendly_name": n.friendly_name} for n in numbers]
