import base64 
import socketio
import requests
import json
from datetime import datetime
from dotenv import load_dotenv
import os

class ICICISimpleWebSocket:
    def __init__(self, api_session_token, app_key):
        self.api_session_token = api_session_token
        self.app_key = app_key
        self.session_token = None
        self.user_id = None
        self.sio = socketio.Client()
        self.setup_events()
    
    def get_websocket_session_token(self):
        """Get the session token from customer details API"""
        try:
            print("Fetching WebSocket session token...")
            
            url = "https://api.icicidirect.com/breezeapi/api/v1/customerdetails"
            payload = json.dumps({
                "SessionToken": self.api_session_token,
                "AppKey": self.app_key
            })
            headers = {
                'Content-Type': 'application/json',
            }
            
            response = requests.request("GET", url, headers=headers, data=payload)
            data = json.loads(response.text)
            
            if data.get("Success") and "session_token" in data["Success"]:
                websocket_session_key = data["Success"]["session_token"]
                print(f"Got WebSocket session key: {websocket_session_key}")
                
                # Decode the session key
                decoded = base64.b64decode(websocket_session_key.encode('ascii')).decode('ascii')
                self.user_id, self.session_token = decoded.split(":")
                print(f"User ID: {self.user_id}")
                print(f"Session Token: {self.session_token}")
                return True
            else:
                print(f"Failed to get session token: {data}")
                return False
                
        except Exception as e:
            print(f"Error getting session token: {e}")
            return False
    
    def setup_events(self):
        """Setup WebSocket event handlers"""
        
        @self.sio.event
        def connect():
            print("WebSocket connected successfully!")
        
        @self.sio.event
        def disconnect():
            print("WebSocket disconnected")
        
        @self.sio.event
        def connect_error(data):
            print(f"Connection error: {data}")
        
        @self.sio.on('stock')
        def on_stock_data(data):
            """Handle incoming stock data"""
            try:
                parsed_data = self.parse_data(data)
                print("📊 Live Data:")
                print(f"   Symbol: {parsed_data.get('symbol', 'N/A')}")
                print(f"   Last Price: {parsed_data.get('last', 'N/A')}")
                print(f"   Change: {parsed_data.get('change', 'N/A')}")
                print(f"   Volume: {parsed_data.get('ltq', 'N/A')}")
                print("-" * 40)
            except Exception as e:
                print(f"Error parsing data: {e}")
                print(f"Raw data: {data}")
    
    def parse_data(self, data):
        """Simple data parser"""
        if not data or not isinstance(data, list) or len(data) == 0:
            return {}
        
        try:
            if len(data) >= 12:
                return {
                    "symbol": data[0],
                    "open": data[1],
                    "last": data[2],
                    "high": data[3],
                    "low": data[4],
                    "change": data[5],
                    "bPrice": data[6],
                    "bQty": data[7],
                    "sPrice": data[8],
                    "sQty": data[9],
                    "ltq": data[10],
                    "avgPrice": data[11]
                }
            else:
                return {"raw_data": data}
        except Exception as e:
            print(f"Parse error: {e}")
            return {"raw_data": data}
    
    def connect_and_stream(self, stock_codes, duration=30):
        """Connect to WebSocket and start streaming"""
        try:
            if not self.get_websocket_session_token():
                print("Failed to get session token")
                return False
            
            print("Connecting to WebSocket...")
            auth = {"user": self.user_id, "token": self.session_token}
            
            self.sio.connect(
                "https://livestream.icicidirect.com", 
                headers={"User-Agent": "python-socketio[client]/socket"}, 
                auth=auth, 
                transports="websocket", 
                wait_timeout=10
            )
            
            if not self.sio.connected:
                print("Failed to connect to WebSocket")
                return False
            
            for stock_code in stock_codes:
                print(f"Subscribing to {stock_code}")
                self.sio.emit('join', stock_code)
            
            print(f"🚀 Streaming for {duration} seconds...")
            self.sio.sleep(duration)
            
            return True
            
        except KeyboardInterrupt:
            print("\nStopping stream...")
        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            try:
                for stock_code in stock_codes:
                    self.sio.emit("leave", stock_code)
                self.sio.emit("disconnect", "transport close")
                self.sio.disconnect()
                print("Disconnected successfully")
            except:
                pass

# Usage
if __name__ == "__main__":
    load_dotenv()
    API_SESSION_TOKEN = "dummy_value"
    APP_KEY = "dummy_value"

    client = ICICISimpleWebSocket(API_SESSION_TOKEN, APP_KEY)
    
    stock_codes = ["4.1!2885"]
    client.connect_and_stream(stock_codes, duration=60)
