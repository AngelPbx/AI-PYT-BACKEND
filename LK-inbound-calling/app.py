from flask import Flask, Response, request

app = Flask(__name__)

@app.route("/voice", methods=['POST'])
def voice():
    print("📞 Incoming call received...")

    response = """
    <?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Dial>
            <Sip>sip:5rlr6yk0zfz.sip.livekit.cloud</Sip>
        </Dial>
    </Response>
    """
    return Response(response, mimetype='text/xml')

@app.route("/", methods=["GET"])
def home():
    return "Flask Server is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
