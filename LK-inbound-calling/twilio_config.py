from twilio.rest import Client

# Twilio credentials
account_sid = 'AC0d8d96ec0d573804abd9648'
auth_token = '48aafb69f4f0724'

client = Client(account_sid, auth_token)

voice_url = 'http://127.0.0.1:5000/voice'  # Update this!

# phone number SID
phone_number_sid = 'PNd326d4f60177dee88ad8891aa6f9745c'  # Replace if incorrect

# Update the phone number
number = client.incoming_phone_numbers(phone_number_sid).update(
    voice_url=voice_url,
    voice_method='POST'
)

print(f"✅ Updated {number.phone_number} to forward calls to {voice_url}")