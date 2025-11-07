"""
Simple script to simulate ESP32 sending data to the server.
Run this to test the server without having the actual ESP32.

Usage: python test_esp32.py
"""
import requests
import time
import random

# Configuration
SERVER_URL = "http://localhost:10000"
DEVICE_ID = "esp32-bed-001"

def send_startup():
    """Send startup package (IDs 1001-1007)"""
    print("📤 Sending startup data...")
    
    data = {
        "CustomName": "Test Bed #1",
        "Status": "online",
        "ActualPartition": "OTA1",
        "VersionFactory": "1.0.0",
        "VersionOTA1": "1.1.0",
        "VersionOTA2": "1.0.5",
        "MAC": "AA:BB:CC:DD:EE:FF"
    }
    
    try:
        response = requests.post(
            f"{SERVER_URL}/esp32/{DEVICE_ID}/startup",
            json=data
        )
        print(f"✅ Startup: {response.json()}")
    except Exception as e:
        print(f"❌ Error: {e}")

def send_sensors():
    """Send sensor data (IDs 1008-1011)"""
    print("📤 Sending sensor data...")
    
    data = {
        "Temperature": round(22 + random.uniform(-2, 2), 1),
        "SDFreeStorage": round(28 + random.uniform(-1, 1), 1),
        "MotorStatus": random.choice(["stopped", "running", "idle"])
    }
    
    try:
        response = requests.post(
            f"{SERVER_URL}/esp32/{DEVICE_ID}/sensors",
            json=data
        )
        print(f"✅ Sensors: {data}")
    except Exception as e:
        print(f"❌ Error: {e}")

def poll_commands():
    """Poll for commands from PWA"""
    print("📥 Polling for commands...")
    
    try:
        response = requests.get(f"{SERVER_URL}/esp32/{DEVICE_ID}/poll")
        data = response.json()
        
        if data.get("commands"):
            print(f"✅ Received {len(data['commands'])} command(s):")
            for cmd in data["commands"]:
                print(f"   - {cmd['command']}: {cmd.get('payload', {})}")
        else:
            print("   No commands pending")
            
        return data.get("commands", [])
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def main():
    print("=" * 60)
    print("🛏️  ESP32 Simulator for Somnomat")
    print("=" * 60)
    print(f"Server: {SERVER_URL}")
    print(f"Device: {DEVICE_ID}")
    print("-" * 60)
    
    # Send startup once
    send_startup()
    print()
    
    # Main loop - simulate ESP32 behavior
    print("Starting main loop (Ctrl+C to stop)...")
    print("-" * 60)
    
    try:
        counter = 0
        while True:
            counter += 1
            print(f"\n[Cycle {counter}]")
            
            # Send sensor data every cycle
            send_sensors()
            
            # Poll for commands every cycle
            commands = poll_commands()
            
            # If we received commands, execute them (just print for now)
            if commands:
                for cmd in commands:
                    print(f"   🎯 Executing: {cmd['command']}")
            
            # Wait 5 seconds before next cycle
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user")
        print("-" * 60)

if __name__ == "__main__":
    main()