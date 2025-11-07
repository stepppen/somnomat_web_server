from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional, List
import random

app = FastAPI(title="Somnomat MVP API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (replace with database later)
latest_data = {}  # device_id -> latest sensor data
pending_commands = {}  # device_id -> list of commands
sleep_data = {}  # device_id -> sleep intervals and events

# ----- Models -----
class ESPStartupData(BaseModel):
    CustomName: str
    Status: str
    ActualPartition: str
    VersionFactory: str
    VersionOTA1: str
    VersionOTA2: str
    MAC: str

class ESPSensorData(BaseModel):
    Temperature: float
    SDFreeStorage: float
    MotorStatus: str

class PWACommand(BaseModel):
    device_id: str
    command: str
    payload: Optional[dict] = {}

# ----- Sleep Data Generation -----
def generate_sleep_intervals(device_id: str, days: int = 7):
    """Generate realistic sleep occupancy intervals for the last N days"""
    intervals = []
    now = datetime.utcnow()
    
    # Generate data for each day going backwards
    for day_offset in range(days):
        day_start = now - timedelta(days=day_offset)
        
        # Random bedtime between 21:30 and 23:30 (UTC)
        bedtime_hour = random.randint(21, 23)
        bedtime_min = random.randint(0, 59) if bedtime_hour < 23 else random.randint(0, 30)
        
        sleep_start = day_start.replace(
            hour=bedtime_hour, 
            minute=bedtime_min, 
            second=0, 
            microsecond=0
        ) - timedelta(days=1)  # previous evening
        
        # Total sleep duration: 6-9 hours
        total_sleep_hours = random.uniform(6.5, 8.5)
        
        # Generate 1-3 sleep segments (awakenings)
        num_segments = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]
        
        remaining_hours = total_sleep_hours
        current_time = sleep_start
        
        for seg in range(num_segments):
            # Duration for this segment
            if seg == num_segments - 1:
                seg_hours = remaining_hours
            else:
                seg_hours = random.uniform(2.0, min(4.0, remaining_hours - 0.5))
            
            seg_start = current_time
            seg_end = seg_start + timedelta(hours=seg_hours)
            
            intervals.append({
                "start": seg_start.isoformat() + "Z",
                "end": seg_end.isoformat() + "Z",
                "duration_min": round(seg_hours * 60, 1)
            })
            
            remaining_hours -= seg_hours
            
            # Add awakening gap (5-30 minutes)
            if seg < num_segments - 1:
                current_time = seg_end + timedelta(minutes=random.randint(5, 30))
            else:
                current_time = seg_end
    
    return intervals

def generate_bed_events(intervals: List[dict]):
    """Generate bed ON/OFF events from sleep intervals"""
    events = []
    
    for interval in intervals:
        # Bed turned ON slightly before sleep start (0-10 min before)
        start_dt = datetime.fromisoformat(interval["start"].replace("Z", ""))
        on_time = start_dt - timedelta(minutes=random.randint(0, 10))
        
        events.append({
            "timestamp": on_time.isoformat() + "Z",
            "event": "OFF->ON"
        })
        
        # Bed turned OFF slightly after sleep end (0-15 min after)
        end_dt = datetime.fromisoformat(interval["end"].replace("Z", ""))
        off_time = end_dt + timedelta(minutes=random.randint(0, 15))
        
        events.append({
            "timestamp": off_time.isoformat() + "Z",
            "event": "ON->OFF"
        })
    
    # Sort by timestamp
    events.sort(key=lambda x: x["timestamp"])
    return events

def generate_occupancy_samples(intervals: List[dict], samples_per_minute: int = 1):
    """Generate raw ADC occupancy samples (lower values = occupied)"""
    samples = []
    
    for interval in intervals:
        start_dt = datetime.fromisoformat(interval["start"].replace("Z", ""))
        end_dt = datetime.fromisoformat(interval["end"].replace("Z", ""))
        
        # Generate samples during occupied period (values < 0.2)
        current = start_dt
        while current < end_dt:
            samples.append({
                "timestamp": current.isoformat() + "Z",
                "adc_value_scaled": round(random.uniform(0.05, 0.18), 3)  # Occupied
            })
            current += timedelta(minutes=1)
        
        # Add some unoccupied samples after (values > 0.2)
        for _ in range(5):
            samples.append({
                "timestamp": current.isoformat() + "Z",
                "adc_value_scaled": round(random.uniform(0.25, 0.85), 3)  # Not occupied
            })
            current += timedelta(minutes=1)
    
    samples.sort(key=lambda x: x["timestamp"])
    return samples

# ----- Health Check -----
@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    }

# ----- ESP32 Endpoints -----
@app.post("/esp32/{device_id}/startup")
def esp32_startup(device_id: str, data: ESPStartupData):
    """ESP32 sends startup info on boot"""
    if device_id not in latest_data:
        latest_data[device_id] = {}
        pending_commands[device_id] = []
        
        # Generate initial sleep data for new devices
        intervals = generate_sleep_intervals(device_id, days=7)
        events = generate_bed_events(intervals)
        samples = generate_occupancy_samples(intervals)
        
        sleep_data[device_id] = {
            "intervals": intervals,
            "events": events,
            "samples": samples
        }
    
    latest_data[device_id].update({
        "custom_name": data.CustomName,
        "status": data.Status,
        "partition": data.ActualPartition,
        "version_factory": data.VersionFactory,
        "version_ota1": data.VersionOTA1,
        "version_ota2": data.VersionOTA2,
        "mac": data.MAC,
        "last_seen": datetime.now().isoformat()
    })
    
    return {"success": True, "message": f"Device {device_id} registered"}

@app.post("/esp32/{device_id}/sensors")
def esp32_sensors(device_id: str, data: ESPSensorData):
    """ESP32 sends sensor readings"""
    if device_id not in latest_data:
        latest_data[device_id] = {}
    
    latest_data[device_id].update({
        "temperature": data.Temperature,
        "sd_free_storage": data.SDFreeStorage,
        "motor_status": data.MotorStatus,
        "last_seen": datetime.now().isoformat()
    })
    
    return {"success": True}

@app.get("/esp32/{device_id}/poll")
def esp32_poll(device_id: str):
    """ESP32 polls for pending commands"""
    if device_id not in pending_commands:
        pending_commands[device_id] = []
    
    # Get all pending commands
    commands = pending_commands[device_id].copy()
    
    # Clear the queue
    pending_commands[device_id] = []
    
    return {
        "device_id": device_id,
        "commands": commands,
        "timestamp": datetime.now().isoformat()
    }

# ----- PWA/Frontend Endpoints -----
@app.get("/devices")
def list_devices():
    """Get all devices with their latest data"""
    devices = []
    for device_id, data in latest_data.items():
        devices.append({
            "device_id": device_id,
            **data
        })
    return {"devices": devices, "total": len(devices)}

@app.get("/devices/{device_id}")
def get_device(device_id: str):
    """Get specific device data"""
    if device_id not in latest_data:
        raise HTTPException(status_code=404, detail="Device not found")
    
    return {
        "device_id": device_id,
        **latest_data[device_id]
    }

@app.post("/commands")
def create_command(cmd: PWACommand):
    """PWA sends command for ESP32"""
    if cmd.device_id not in pending_commands:
        pending_commands[cmd.device_id] = []
    
    command_obj = {
        "command": cmd.command,
        "payload": cmd.payload,
        "created_at": datetime.now().isoformat()
    }
    
    pending_commands[cmd.device_id].append(command_obj)
    
    # Echo back the command
    return {
        "success": True,
        "echo": command_obj,
        "message": f"Command '{cmd.command}' queued for {cmd.device_id}"
    }

@app.get("/commands/{device_id}")
def get_pending_commands(device_id: str):
    """Check pending commands for a device"""
    if device_id not in pending_commands:
        return {"device_id": device_id, "commands": []}
    
    return {
        "device_id": device_id,
        "commands": pending_commands[device_id]
    }

# ----- Sleep Data Endpoints -----
@app.get("/sleep/{device_id}/intervals")
def get_sleep_intervals(device_id: str, days: int = 7):
    """Get sleep occupancy intervals for a device"""
    if device_id not in sleep_data:
        # Generate on-demand if not exists
        intervals = generate_sleep_intervals(device_id, days)
        sleep_data[device_id] = {
            "intervals": intervals,
            "events": generate_bed_events(intervals),
            "samples": generate_occupancy_samples(intervals)
        }
    
    return {
        "device_id": device_id,
        "intervals": sleep_data[device_id]["intervals"],
        "total": len(sleep_data[device_id]["intervals"])
    }

@app.get("/sleep/{device_id}/events")
def get_bed_events(device_id: str):
    """Get bed power ON/OFF events"""
    if device_id not in sleep_data:
        raise HTTPException(status_code=404, detail="No sleep data for device")
    
    return {
        "device_id": device_id,
        "events": sleep_data[device_id]["events"],
        "total": len(sleep_data[device_id]["events"])
    }

@app.get("/sleep/{device_id}/samples")
def get_occupancy_samples(device_id: str):
    """Get raw occupancy sensor samples"""
    if device_id not in sleep_data:
        raise HTTPException(status_code=404, detail="No sleep data for device")
    
    return {
        "device_id": device_id,
        "samples": sleep_data[device_id]["samples"],
        "total": len(sleep_data[device_id]["samples"])
    }

@app.get("/sleep/{device_id}/summary")
def get_sleep_summary(device_id: str):
    """Get comprehensive sleep summary for dashboard"""
    # Debug logging
    print(f"[DEBUG] Summary requested for device: {device_id}")
    print(f"[DEBUG] Available devices: {list(latest_data.keys())}")
    print(f"[DEBUG] Sleep data devices: {list(sleep_data.keys())}")
    
    # Check if device exists
    if device_id not in latest_data:
        available = list(latest_data.keys())
        raise HTTPException(
            status_code=404, 
            detail=f"Device {device_id} not found. Available devices: {available}. Make sure the ESP32 simulator is running."
        )
    
    # Generate sleep data on-demand if not exists
    if device_id not in sleep_data:
        print(f"[DEBUG] Generating sleep data for {device_id}")
        intervals = generate_sleep_intervals(device_id, days=7)
        sleep_data[device_id] = {
            "intervals": intervals,
            "events": generate_bed_events(intervals),
            "samples": generate_occupancy_samples(intervals)
        }
        print(f"[DEBUG] Generated {len(intervals)} intervals")
    
    data = sleep_data[device_id]
    intervals = data["intervals"]
    
    # Calculate statistics
    total_sleep_min = sum(iv["duration_min"] for iv in intervals)
    avg_sleep_per_night = total_sleep_min / 7  # assuming 7 days
    
    # Group by day and count awakenings
    from collections import defaultdict
    by_day = defaultdict(list)
    for iv in intervals:
        day = iv["start"][:10]  # YYYY-MM-DD
        by_day[day].append(iv)
    
    awakenings_per_day = {day: len(ivs) - 1 for day, ivs in by_day.items()}
    avg_awakenings = sum(awakenings_per_day.values()) / len(awakenings_per_day) if by_day else 0
    
    return {
        "device_id": device_id,
        "summary": {
            "total_sleep_hours": round(total_sleep_min / 60, 1),
            "avg_sleep_per_night_hours": round(avg_sleep_per_night / 60, 1),
            "total_nights": len(by_day),
            "total_intervals": len(intervals),
            "avg_awakenings_per_night": round(avg_awakenings, 1),
            "awakenings_by_day": awakenings_per_day
        },
        "intervals": intervals,
        "events": data["events"]
    }

# ----- Debug Endpoint -----
@app.get("/debug")
def debug_state():
    """See all stored data (for development only)"""
    return {
        "latest_data": latest_data,
        "pending_commands": pending_commands,
        "sleep_data_keys": list(sleep_data.keys()),
        "total_devices": len(latest_data),
        "total_sleep_devices": len(sleep_data)
    }

@app.get("/test")
def test_endpoint():
    """Simple test endpoint to verify API is running"""
    return {
        "status": "API is running!",
        "timestamp": datetime.now().isoformat(),
        "devices_count": len(latest_data),
        "devices": list(latest_data.keys())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)