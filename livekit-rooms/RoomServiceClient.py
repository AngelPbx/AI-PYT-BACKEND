import asyncio
import os
import logging
from livekit import api
from dotenv import load_dotenv
from livekit.api import CreateRoomRequest
import aiohttp

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_environment():
    """Load and validate environment variables"""
    # Ensure .env file is loaded
    if not load_dotenv():
        raise EnvironmentError("Could not load .env file")

    # Get required environment variables
<<<<<<< HEAD
    host = os.getenv('LIVEKIT_URL')
=======
    host = os.getenv('LIVEKIT_HOST')
>>>>>>> arbaz
    api_key = os.getenv('LIVEKIT_API_KEY')
    secret = os.getenv('LIVEKIT_API_SECRET')

    # Validate environment variables
    missing_vars = []
    if not host:
        missing_vars.append('LIVEKIT_HOST')
    if not api_key:
        missing_vars.append('LIVEKIT_API_KEY')
    if not secret:
        missing_vars.append('LIVEKIT_API_SECRET')

    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

    return host, api_key, secret

async def manage_rooms():
    try:
        # Load environment variables
        host, api_key, secret = load_environment()
        
        # Ensure host URL is properly formatted
        if not host.startswith(('http://', 'https://')):
            host = f'https://{host}'

        # Remove 'wss://' if present in the host URL
        host = host.replace('wss://', '')

        logger.info(f"Connecting to LiveKit server: {host}")
        logger.debug(f"Using API Key: {api_key[:8]}...")
        
        # Initialize LiveKit API without debug parameter
        livekit_api = api.LiveKitAPI(
            url=host,
            api_key=api_key,
            api_secret=secret
        )

        # Create room request
        room_request = CreateRoomRequest(
            name="test-room",
            empty_timeout=10 * 60,
            max_participants=20
        )

        logger.debug("Attempting to create room...")
        room = await livekit_api.room.create_room(room_request)
        logger.info(f"Room created: {room.name}")

        # List all rooms
        logger.info("Listing all rooms:")
        rooms = await livekit_api.room.list_rooms()
        for r in rooms:
            logger.info(f"- Room: {r.name}, Participants: {r.num_participants}")

        # Delete the room
        logger.debug("Attempting to delete room...")
        await livekit_api.room.delete_room("test-room")
        logger.info("Room deleted.")

    except EnvironmentError as e:
        logger.error(f"Environment Error: {str(e)}")
        logger.error("Please check your .env file and ensure all required variables are set")
    except aiohttp.ClientResponseError as e:
        logger.error(f"Authentication error: {str(e)}")
        logger.error("Please verify your LiveKit credentials in the .env file")
        logger.error(f"Host: {host}")
        logger.error(f"API Key length: {len(api_key) if api_key else 0}")
        logger.error(f"API Secret length: {len(secret) if secret else 0}")
        logger.error(f"Response status: {e.status}")
        logger.error(f"Response message: {e.message}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
    finally:
        if 'livekit_api' in locals():
            await livekit_api.aclose()

if __name__ == "__main__":
    asyncio.run(manage_rooms())