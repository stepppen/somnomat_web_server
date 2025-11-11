from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import random
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import asyncio
import math

load_dotenv()
app = FastAPI(title="MVP 1.1 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# In-memory storage; later db
latest_data = {}  
pending_commands = {} 
sleep_data = {} 
aggregated_occupancy = {} 

push_task = None

class ESPStartupData(BaseModel):
    SDFreeStorage: float
    CustomName: str
    Status: str
    ActualPartition: str
    VersionFactory: str
    VersionOTA1: str
    VersionOTA2: str
    MAC: str

class ESPSensorData(BaseModel):
    Temperature: float
    MotorStatus: str
    Occupancy: bool
    Speed: float
    Acceleration: float
    Intensity: float

class PWACommand(BaseModel):
    device_id: str
    command: str
    payload: Optional[dict] = {}

class SleepMetricsCache:
    def __init__(self):
        self.last_full_compute = None
        self.cached_intervals = []
        
sleep_cache = SleepMetricsCache()


#aggregation function occupancy
def aggregate_occupancy(data: bool):
    aggregated_occupancy.append({
        "occupied": data,
        "timestamp": datetime.now().isoformat()
        })

def summarize_minute(batch):
    """Return true if more than half are occ"""
    if not batch:
        return None
    occupied_count = sum(1 for x in batch if x["occupied"])
    occupancy_value = occupied_count >= len(batch) / 2
    return occupancy_value

async def push_aggregated_data(device_id: str):
    """Runs every min"""
    while True:
        await asyncio.sleep(60)
        if not aggregated_occupancy:
            continue

        batch = aggregated_occupancy.copy()
        aggregated_occupancy.clear()

        #one bool for whole minute
        occupied = summarize_minute(batch)

        #execute functions that calculate different values off of occupancy
        metrics = compute_sleep_metrics(device_id, occupied)

        record = {
            "device_id": device_id,
            "created_at": datetime.now().isoformat(),
            "occupied": occupied,
            "sleep_consistency": metrics.get("consistency_score", 0),
            "bedtime_consistency": metrics.get("consistency_sd_minutes", 0),
            "bed_use": metrics.get("bed_on_coverage_pct", 0),
            "daily_occupancy": metrics.get("avg_sleep_per_night", 0), 
            "total_intervals": metrics.get("total_intervals", 0),
            "total_nights": metrics.get("total_nights", 0),
            "avg_sleep_per_night": metrics.get("avg_sleep_per_night", 0),
        }


        try:
            supabase.table("sleep_dashboard").insert(record).execute()
            print(f"Pushed aggregated data for {device_id} at {datetime.now()}")
        except Exception as e:
            print(f"Error pushing data: {e}")


#calculating sleep metrics
def compute_sleep_metrics(device_id: str, occupied: bool) -> Dict:
    """
    Compute sleep metrics based on current occupancy and historical data.
    """
    try:
        current_record = {
            "device_id": device_id,
            "timestamp": datetime.now().isoformat(),
            "occupied": occupied
        }
        supabase.table("raw_occupancy").insert(current_record).execute()
    except Exception as e:
        print(f"Error inserting raw occupancy: {e}")
    

    now = datetime.now()
    seven_days_ago = now - timedelta(days=7)
    
    try:
        response = supabase.table("raw_occupancy")\
            .select("*")\
            .eq("device_id", device_id)\
            .gte("timestamp", seven_days_ago.isoformat())\
            .order("timestamp")\
            .execute()
        
        occupancy_rows = response.data
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        occupancy_rows = []
    
    occ_intervals = build_occupancy_intervals(occupancy_rows)
    
    metrics = {
        "total_sleep_minutes": compute_total_sleep(occ_intervals),
        "total_nights": count_nights(occ_intervals),
        "total_intervals": len(occ_intervals),
        "avg_sleep_per_night": compute_avg_sleep_per_night(occ_intervals),
        "avg_awakenings_per_night": compute_avg_awakenings(occ_intervals),
        "awakenings_last_night": compute_awakenings_last_night(occ_intervals),
        "consistency_score": compute_consistency_score(occ_intervals),
        "consistency_sd_minutes": compute_consistency_sd(occ_intervals)
    }
    
    return metrics

def build_occupancy_intervals(rows: List[Dict]) -> List[Dict]:
    """
    bools occupancy -> continuous intervals.
    """
    if not rows:
        return []
    
    samples = sorted(rows, key=lambda x: x["timestamp"])
    MIN_SEG_SEC = 120 
    
    intervals = []
    in_occ = False
    current_start = None
    last_timestamp = None
    
    for sample in samples:
        timestamp = datetime.fromisoformat(sample["timestamp"].replace("Z", ""))
        occupied = sample.get("occupied", False)
        
        if occupied and not in_occ:
            in_occ = True
            current_start = timestamp
        elif not occupied and in_occ:
            in_occ = False
            if last_timestamp and (last_timestamp - current_start).total_seconds() >= MIN_SEG_SEC:
                intervals.append({
                    "start": current_start,
                    "end": last_timestamp,
                    "duration_min": (last_timestamp - current_start).total_seconds() / 60
                })
            current_start = None
        
        last_timestamp = timestamp
    
    if in_occ and current_start and last_timestamp:
        if (last_timestamp - current_start).total_seconds() >= MIN_SEG_SEC:
            intervals.append({
                "start": current_start,
                "end": last_timestamp,
                "duration_min": (last_timestamp - current_start).total_seconds() / 60
            })
    
    return intervals


def compute_total_sleep(intervals: List[Dict]) -> float:
    """Total minutes of occupancy across all intervals"""
    return sum(iv["duration_min"] for iv in intervals)


def count_nights(intervals: List[Dict]) -> int:
    """Count unique days with occupancy data"""
    if not intervals:
        return 0
    
    days = set()
    for iv in intervals:
        day_key = iv["start"].date().isoformat()
        days.add(day_key)
    
    return len(days)


def compute_avg_sleep_per_night(intervals: List[Dict]) -> float:
    """Average sleep duration per night"""
    if not intervals:
        return 0.0
    
    by_day = {}
    for iv in intervals:
        day = iv["start"].date().isoformat()
        if day not in by_day:
            by_day[day] = 0
        by_day[day] += iv["duration_min"]
    
    if not by_day:
        return 0.0
    
    return sum(by_day.values()) / len(by_day)


def compute_avg_awakenings(intervals: List[Dict]) -> float:
    """
    Average number of awakenings per night.
    """
    if not intervals:
        return 0.0
    
    by_day = {}
    for iv in intervals:
        day = iv["start"].date().isoformat()
        if day not in by_day:
            by_day[day] = []
        by_day[day].append(iv)
    
    if not by_day:
        return 0.0
    
    total_awakenings = 0
    for day, day_intervals in by_day.items():
        awakenings = max(0, len(day_intervals) - 1)
        total_awakenings += awakenings
    
    return total_awakenings / len(by_day)


def compute_awakenings_last_night(intervals: List[Dict]) -> int:
    """Number of awakenings on the most recent night"""
    if not intervals:
        return 0
    
    latest_day = max(iv["start"].date() for iv in intervals)
    day_intervals = [iv for iv in intervals if iv["start"].date() == latest_day]
    
    return max(0, len(day_intervals) - 1)


def compute_consistency_score(intervals: List[Dict]) -> int:
    """
    Bedtime consistency score (0-100).
    Based on circular standard deviation of first sleep interval each night.
    """
    sd_min = compute_consistency_sd(intervals)
    if sd_min is None:
        return 0
    
    clamp_180 = min(180, max(0, sd_min))
    score = round(100 * (1 - clamp_180 / 180))
    
    return score


def compute_consistency_sd(intervals: List[Dict]) -> Optional[float]:
    """
    Circular standard deviation of bedtime in minutes.
    """
    if not intervals:
        return None
    
    by_day = {}
    for iv in intervals:
        day = iv["start"].date().isoformat()
        if day not in by_day:
            by_day[day] = []
        by_day[day].append(iv)
    
    first_starts = []
    for day in sorted(by_day.keys()):
        day_intervals = sorted(by_day[day], key=lambda x: x["start"])
        first = next((iv for iv in day_intervals if iv["duration_min"] >= 2), None)
        if first:
            first_starts.append(first["start"])
    
    if len(first_starts) < 2:
        return 0.0
    
    mins = [
        dt.hour * 60 + dt.minute + dt.second / 60
        for dt in first_starts
    ]
    
    rad = [m / (24 * 60) * 2 * math.pi for m in mins]
    
    C = sum(math.cos(t) for t in rad) / len(rad)
    S = sum(math.sin(t) for t in rad) / len(rad)
    R = math.sqrt(C * C + S * S)
    
    if R == 0:
        sd_min = 12 * 60  # max spread
    else:
        s = math.sqrt(-2 * math.log(R))
        sd_min = s / (2 * math.pi) * 24 * 60
    
    return round(sd_min)

# #mock sleep data-------
# def generate_sleep_intervals(device_id: str, days: int = 7):
#     """Generating sleep occupancy data"""
#     intervals = []
#     now = datetime.now()
    
#     #data for each day
#     for day_offset in range(days):
#         day_start = now - timedelta(days=day_offset)
#         bedtime_hour = random.randint(21, 23)
#         bedtime_min = random.randint(0, 59) if bedtime_hour < 23 else random.randint(0, 30)
        
#         sleep_start = day_start.replace(
#             hour=bedtime_hour, 
#             minute=bedtime_min, 
#             second=0, 
#             microsecond=0
#         ) - timedelta(days=1) 
        
#         #6-9 hours
#         total_sleep_hours = random.uniform(6.5, 8.5)
#         num_segments = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]
#         remaining_hours = total_sleep_hours
#         current_time = sleep_start
        
#         for seg in range(num_segments):
#             if seg == num_segments - 1:
#                 seg_hours = remaining_hours
#             else:
#                 seg_hours = random.uniform(2.0, min(4.0, remaining_hours - 0.5))
            
#             seg_start = current_time
#             seg_end = seg_start + timedelta(hours=seg_hours)
            
#             intervals.append({
#                 "start": seg_start.isoformat() + "Z",
#                 "end": seg_end.isoformat() + "Z",
#                 "duration_min": round(seg_hours * 60, 1)
#             })
            
#             remaining_hours -= seg_hours
            
#             # awakening gap: 5-30 min
#             if seg < num_segments - 1:
#                 current_time = seg_end + timedelta(minutes=random.randint(5, 30))
#             else:
#                 current_time = seg_end
    
#     return intervals

# def generate_bed_events(intervals: List[dict]):
#     """Generate bed ON/OFF events from sleep intervals"""
#     events = []
    
#     for interval in intervals:
#         start_dt = datetime.fromisoformat(interval["start"].replace("Z", ""))
#         on_time = start_dt - timedelta(minutes=random.randint(0, 10))
        
#         events.append({
#             "timestamp": on_time.isoformat() + "Z",
#             "event": "OFF->ON"
#         })
        
#         end_dt = datetime.fromisoformat(interval["end"].replace("Z", ""))
#         off_time = end_dt + timedelta(minutes=random.randint(0, 15))
        
#         events.append({
#             "timestamp": off_time.isoformat() + "Z",
#             "event": "ON->OFF"
#         })
    
#     events.sort(key=lambda x: x["timestamp"])
#     return events

# def generate_occupancy_samples(intervals: List[dict], samples_per_minute: int = 1):
#     """Generate raw ADC occupancy samples (lower values = occupied)"""
#     samples = []
    
#     for interval in intervals:
#         start_dt = datetime.fromisoformat(interval["start"].replace("Z", ""))
#         end_dt = datetime.fromisoformat(interval["end"].replace("Z", ""))
        
#         current = start_dt
#         while current < end_dt:
#             samples.append({
#                 "timestamp": current.isoformat() + "Z",
#                 "adc_value_scaled": round(random.uniform(0.05, 0.18), 3)  
#             })
#             current += timedelta(minutes=1)
        
#         for _ in range(5):
#             samples.append({
#                 "timestamp": current.isoformat() + "Z",
#                 "adc_value_scaled": round(random.uniform(0.25, 0.85), 3) 
#             })
#             current += timedelta(minutes=1)
    
#     samples.sort(key=lambda x: x["timestamp"])
#     return samples



# ----- ESP32 Endpoints -----
@app.post("/esp32/{device_id}/startup")
def esp32_startup(device_id: str, data: ESPStartupData):
    """ESP32 -> sends info on boot"""
    if device_id not in latest_data:
        latest_data[device_id] = {}
        pending_commands[device_id] = []
        
    
    latest_data[device_id].update({
        "custom_name": data.CustomName,
        "status": data.Status,
        "partition": data.ActualPartition,
        "version_factory": data.VersionFactory,
        "version_ota1": data.VersionOTA1,
        "version_ota2": data.VersionOTA2,
        "sd_free_storage": data.SDFreeStorage,
        "mac": data.MAC,
        "last_seen": datetime.now().isoformat()
    })
    device_record = {
        "device_id": device_id,  
        "name": data.CustomName,
        "custom_name": data.CustomName,
        "status": data.Status,
        "partition": data.ActualPartition,
        "version_factory": data.VersionFactory,
        "version_ota1": data.VersionOTA1,
        "version_ota2": data.VersionOTA2,
        "sd_free_storage": data.SDFreeStorage,
        "mac": data.MAC,
        "last_seen": datetime.now().isoformat()
    }
    try:
        supabase.table("devices").upsert(device_record).execute()
    except Exception as e:
        print(f"Error upserting device: {e}")
    
    return {"success": True, "message": f"Device {device_id} registered"}

@app.post("/esp32/{device_id}/sensors")
def esp32_sensors(device_id: str, data: ESPSensorData):
    """ESP32 -> sends sensor readings"""
    global push_task
    if device_id not in latest_data:
        latest_data[device_id] = {}
    
    latest_data[device_id].update({
        "temperature": data.Temperature,
        "motor_status": data.MotorStatus,
        "occupancy": data.Occupancy,
        "speed": data.Speed,
        "acceleration": data.Acceleration,
        "intensity": data.Intensity,
        "last_seen": datetime.now().isoformat()
    })

    aggregate_occupancy(data.Occupancy)
    if push_task is None or push_task.done():
            push_task = asyncio.create_task(push_aggregated_data())

    
    return {"success": True}

@app.get("/esp32/{device_id}/poll")
def esp32_poll(device_id: str):
    """ESP32 -> poll for pending commands"""
    if device_id not in pending_commands:
        pending_commands[device_id] = []
    
    # Get current commands
    commands = pending_commands[device_id].copy()
    pending_commands[device_id] = []
    
    return {
        "device_id": device_id,
        "commands": commands,
        "timestamp": datetime.now().isoformat()
    }

# ----- PWA endpoints -----
@app.get("/devices")
def list_devices():
    """Get all devices with data"""
    devices = []
    for device_id, data in latest_data.items():
        devices.append({
            "device_id": device_id,
            **data
        })
    return {"devices": devices, "total": len(devices)}

#where the real-time values should be fetched from specific ESP e.g. occupied
#Buffers data in memory for aggregation
@app.get("/devices/{device_id}")
def get_device(device_id: str):
    """Get specific device data"""

    # if device_id not in latest_data:
    #     raise HTTPException(status_code=404, detail="Device not found")
    
    response = supabase.table("devices").select("*").eq("id", device_id).execute()
    
    # echo
    return response

@app.post("/commands")
def create_command(cmd: PWACommand):
    """create command for ESP32"""
    if cmd.device_id not in pending_commands:
        pending_commands[cmd.device_id] = []
    
    command_obj = {
        "command": cmd.command,
        "payload": cmd.payload,
        "created_at": datetime.now().isoformat()
    }
    
    pending_commands[cmd.device_id].append(command_obj)
    
    # echo
    return {
        "success": True,
        "echo": command_obj,
        "message": f"Command '{cmd.command}' queued for {cmd.device_id}"
    }

# ----- Sleep Data individual endpoints (not needed for now as we have summary endpoint)
# @app.get("/sleep/{device_id}/intervals")
# def get_sleep_intervals(device_id: str, days: int = 7):
#     """Get sleep occupancy intervals for a device"""
#     if device_id not in sleep_data:
#         # Generate if not exists
#         intervals = generate_sleep_intervals(device_id, days)
#         sleep_data[device_id] = {
#             "intervals": intervals,
#             "events": generate_bed_events(intervals),
#             "samples": generate_occupancy_samples(intervals)
#         }
    
#     return {
#         "device_id": device_id,
#         "intervals": sleep_data[device_id]["intervals"],
#         "total": len(sleep_data[device_id]["intervals"])
#     }

# @app.get("/sleep/{device_id}/events")
# def get_bed_events(device_id: str):
#     if device_id not in sleep_data:
#         raise HTTPException(status_code=404, detail="No sleep data for device")
    
#     return {
#         "device_id": device_id,
#         "events": sleep_data[device_id]["events"],
#         "total": len(sleep_data[device_id]["events"])
#     }

# @app.get("/sleep/{device_id}/samples")
# def get_occupancy_samples(device_id: str):
#     """Get raw occupancy sensor samples"""
#     if device_id not in sleep_data:
#         raise HTTPException(status_code=404, detail="No sleep data for device")
    
#     return {
#         "device_id": device_id,
#         "samples": sleep_data[device_id]["samples"],
#         "total": len(sleep_data[device_id]["samples"])
#     }



#where the aggregated values should be fetched from DB 
@app.get("/sleep/{device_id}/summary")
def get_sleep_summary(device_id: str):
    """Get full sleep summary for dashboard"""
    # debuggin
    print(f"[DEBUG] Summary requested for device: {device_id}")
    print(f"[DEBUG] Available devices: {list(latest_data.keys())}")
    print(f"[DEBUG] Sleep data devices: {list(sleep_data.keys())}")
    
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
    total_sleep_min = sum(iv["duration_min"] for iv in intervals)
    avg_sleep_per_night = total_sleep_min / 7 
    
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
    """See all stored data (for dev only)"""
    return {
        "status": "API is running",
        "timestamp": datetime.now().isoformat(),
        "latest_data": latest_data,
        "pending_commands": pending_commands,
        "sleep_data_keys": list(sleep_data.keys()),
        "total_devices": len(latest_data),
        "total_sleep_devices": len(sleep_data)
    }

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "status": "API is running",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "test": "/test",
            "devices": "/devices",
            "esp32_startup": "/esp32/{device_id}/startup",
            "esp32_sensors": "/esp32/{device_id}/sensors",
            "esp32_poll": "/esp32/{device_id}/poll"
        }
    }


#main guard
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)