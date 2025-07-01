from twilio.rest import Client

# Twilio credentials
account_sid = 'AC0d8d96e8bec0d573804a160bd96483b3'
auth_token = '480fa77f56988afb69fb3314f0724c98'

client = Client(account_sid, auth_token)

# Correct voice URL
voice_url = 'https://a8a9-14-194-174-182.ngrok-free.app/voice'

# Update the phone number
number = client.incoming_phone_numbers('PNd326d4f60177dee88ad8891aa6f9745c').update(
    voice_url=voice_url,
    voice_method='POST'
)

print(f"✅ Updated {number.phone_number} to forward calls to {voice_url}")
