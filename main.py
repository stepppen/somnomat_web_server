from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from datetime import datetime, timedelta, time, timezone
from typing import Optional, List, Dict, Literal
from contextlib import asynccontextmanager
from zoneinfo import ZoneInfo
import random
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import asyncio
import math

active_push_tasks: Dict[str, asyncio.Task] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    for task in active_push_tasks.values():
        task.cancel()


load_dotenv()
app = FastAPI(title="MVP 1.2 API", lifespan=lifespan)

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

# In-memory storage for aggregation
latest_data = {}  
pending_commands = {} 
sleep_data = {} 
aggregated_occupancy_binary = [] 
occ_intervals = []
device_metadata = {}

default_metrics = {
    "consistency_score": 0,
    "consistency_sd_minutes": 0,
    "bed_on_coverage_pct": 0,
    "avg_sleep_per_night": 0,
    "total_intervals": 0,
    "total_nights": 0,
    "avg_awakenings": 0
}


push_task = None

class ESPStartupData(BaseModel):
    CustomName: str
    Status: str
    ActualPartition: int
    VersionFactory: str
    VersionOTA1: str
    VersionOTA2: str
    MAC: str
    Temperature: float
    SDFreeStorage: float

class ESPSensorData(BaseModel):
    Speed: float
    Acceleration: float
    Vibration: int
    Intensity: int
    MotorStatus: int
    Safety: int
    Occupancy: bool
    OccupancyRaw: float
    OccupancyBinary: int

class PWACommand(BaseModel):
    device_id: str
    command: str
    payload: Optional[dict] = {}

class SleepMetricsCache:
    def __init__(self):
        self.last_full_compute = None
        self.cached_intervals = []

class UserSettingsUpdate(BaseModel):
    bed_time: str
    wake_up_time: str
    bed_time_tolerance: int
    wake_up_tolerance: int
    
    @field_validator('bed_time', 'wake_up_time')
    @classmethod
    def validate_time_format(cls, v):
        try:
            datetime.strptime(v, '%H:%M')
            return v
        except ValueError:
            raise ValueError('Time must be in HH:MM format')
    
    @field_validator('bed_time_tolerance', 'wake_up_tolerance')
    @classmethod
    def validate_tolerance(cls, v):
        if v < 0:
            raise ValueError('Tolerance must be non-negative')
        return v
        
sleep_cache = SleepMetricsCache()


#aggregation function occupancy
def aggregate_occupancy_binary(data: int):
    aggregated_occupancy_binary.append({
        "occupancy_binary": data,
        "timestamp": datetime.now().isoformat()
        })

def summarize_minute(batch):
    """Return true if more than half are occ"""
    if not batch:
        return None
    occupied_count = sum(1 for x in batch if x["occupancy_binary"] == 1 )
    occupancy_value = occupied_count >= len(batch) / 2
    return occupancy_value

async def push_aggregated_data(device_id: str):
    """Runs every min"""
    while True:
        await asyncio.sleep(60)
        if not aggregated_occupancy_binary:
            continue

        batch = aggregated_occupancy_binary.copy()
        aggregated_occupancy_binary.clear()

        # One bool for whole minute
        occupied = summarize_minute(batch)

        # Check if device exists before computing metrics
        # try:
        #     result = supabase.table("devices").select("id").eq("id", int(device_id)).execute()
        #     if not result.data:
        #         print(f"Device {device_id} not registered. Skipping metric computation.")
        #         continue
        # except Exception as e:
        #     print(f"Error checking device in push_aggregated_data: {e}")
        #     continue

        try:
            current_record = {
                "device_id": int(device_id),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "occupied": occupied,
            }
            supabase.table("raw_occupancy").insert(current_record).execute()

        except Exception as e:
            print(f"Error inserting raw occupancy: {e}")
            return default_metrics

# need to rethink this
def build_occupancy_intervals(rows: List[Dict]) -> List[Dict]:
    """
    bools occupancy -> continuous intervals.
    Fixed to handle UTC strings from Supabase robustly.
    """
    if not rows:
        return []
    
    # Sort by created_at
    samples = sorted(rows, key=lambda x: x["created_at"]) 
    MIN_SEG_SEC = 120 
    
    intervals = []
    in_occ = False
    current_start = None
    last_timestamp = None
    
    for sample in samples:
        # Robust UTC timestamp parsing
        ts_str = sample["created_at"]
        try:
            # Replace Z with +00:00 to ensure Python reads it as UTC
            timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            # Fallback
            timestamp = datetime.fromisoformat(ts_str)
            
        # Ensure it is timezone aware (UTC)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
            
        occupied = sample.get("occupied", False)

        if occupied and not in_occ:
            in_occ = True
            current_start = timestamp
        elif not occupied and in_occ:
            in_occ = False
            # Check duration
            if current_start and last_timestamp:
                duration_sec = (last_timestamp - current_start).total_seconds()
                if duration_sec >= MIN_SEG_SEC:
                    intervals.append({
                        "start": current_start,
                        "end": last_timestamp,
                        "duration_min": duration_sec / 60
                    })
            current_start = None
        
        last_timestamp = timestamp
    
    # Handle the final interval (e.g., if user is still in bed when query runs)
    if in_occ and current_start and last_timestamp:
        duration_sec = (last_timestamp - current_start).total_seconds()
        if duration_sec >= MIN_SEG_SEC:
            intervals.append({
                "start": current_start,
                "end": last_timestamp,
                "duration_min": duration_sec / 60
            })
    
    return intervals

def filter_intervals_by_date_range(intervals: List[Dict], start_date: datetime, end_date: datetime) -> List[Dict]:
    """Filter intervals that fall within the date range"""
    filtered = []
    for iv in intervals:
        interval_start = iv["start"] if isinstance(iv["start"], datetime) else datetime.fromisoformat(str(iv["start"]).replace("Z", ""))
        
        # Check if interval overlaps with our date range
        if interval_start.date() >= start_date.date() and interval_start.date() <= end_date.date():
            filtered.append(iv)
    
    return filtered

def convert_to_cet(dt: datetime) -> datetime:
    """Convert a datetime to CET/CEST timezone"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(ZoneInfo("Europe/Zurich"))

def compute_sleep_metrics_for_range(intervals: List[Dict], start_date: datetime, end_date: datetime) -> Dict:
    """
    Compute comprehensive sleep metrics for a specific date range
    """
    if not intervals:
        return {
            "sleep_duration_min": 0,
            "sleep_duration_hours": 0,
            "bed_activity_min": 0,
            "bed_activity_hours": 0,
            "consistency_score": 0,
            "consistency_sd_minutes": 0,
            "total_intervals": 0,
            "avg_awakenings": 0,
            "wake_up_time": None,
            "bed_time": None,
            "time_in_bed_min": 0,
            "time_in_bed_hours": 0,
            "nights_count": 0
        }
    
    # Total sleep duration
    total_sleep_min = sum(iv["duration_min"] for iv in intervals)
    
    # Group intervals by "sleep night" - intervals that belong to the same sleep session
    sleep_sessions = []
    current_session = []
    
    sorted_intervals = sorted(intervals, key=lambda x: x["start"] if isinstance(x["start"], datetime) else datetime.fromisoformat(str(x["start"]).replace("Z", "")))
    
    for iv in sorted_intervals:
        interval_start = iv["start"] if isinstance(iv["start"], datetime) else datetime.fromisoformat(str(iv["start"]).replace("Z", ""))
        
        if not current_session:
            current_session.append(iv)
        else:
            last_interval = current_session[-1]
            last_end = last_interval["end"] if isinstance(last_interval["end"], datetime) else datetime.fromisoformat(str(last_interval["end"]).replace("Z", ""))
            
            # If gap is more than 8 hours, it's a new sleep session
            gap_hours = (interval_start - last_end).total_seconds() / 3600
            if gap_hours > 8:
                sleep_sessions.append(current_session)
                current_session = [iv]
            else:
                current_session.append(iv)
    
    if current_session:
        sleep_sessions.append(current_session)
    
    nights_count = len(sleep_sessions)
    
    # Bed activity (time motor was ON)
    bed_activity_min = total_sleep_min
    
    # Calculate awakenings (gaps between intervals within same session)
    total_awakenings = 0
    for session in sleep_sessions:
        awakenings = max(0, len(session) - 1)
        total_awakenings += awakenings
    
    avg_awakenings = total_awakenings / nights_count if nights_count > 0 else 0
    
    # Bedtime consistency (using first interval of each session)
    bedtimes_minutes = []
    waketimes_minutes = []
    
    for session in sleep_sessions:
        if session:
            # First interval start = bedtime (CONVERT TO CET!)
            first_start = session[0]["start"] if isinstance(session[0]["start"], datetime) else datetime.fromisoformat(str(session[0]["start"]).replace("Z", ""))
            first_start_cet = convert_to_cet(first_start)
            
            # Use circular time representation for bedtimes (handle midnight wraparound)
            # Convert to minutes since midnight IN CET
            bedtime_min = first_start_cet.hour * 60 + first_start_cet.minute
            bedtimes_minutes.append(bedtime_min)
            
            # Last interval end = wake time (CONVERT TO CET!)
            last_end = session[-1]["end"] if isinstance(session[-1]["end"], datetime) else datetime.fromisoformat(str(session[-1]["end"]).replace("Z", ""))
            last_end_cet = convert_to_cet(last_end)
            waketime_min = last_end_cet.hour * 60 + last_end_cet.minute
            waketimes_minutes.append(waketime_min)
    
    # Circular standard deviation for bedtime consistency
    if len(bedtimes_minutes) >= 2:
        # Use circular statistics for times that wrap around midnight
        rad = [m / (24 * 60) * 2 * math.pi for m in bedtimes_minutes]
        C = sum(math.cos(t) for t in rad) / len(rad)
        S = sum(math.sin(t) for t in rad) / len(rad)
        R = math.sqrt(C * C + S * S)
        
        if R == 0:
            consistency_sd = 12 * 60
        else:
            s = math.sqrt(-2 * math.log(R))
            consistency_sd = s / (2 * math.pi) * 24 * 60
        
        consistency_score = max(0, 100 - (consistency_sd / 180 * 100))
    else:
        consistency_sd = 0
        consistency_score = 100 if len(bedtimes_minutes) == 1 else 0
    
    # Calculate average bedtime using circular mean (IN CET!)
    avg_bedtime = None
    if bedtimes_minutes:
        # Convert to radians
        rad = [m / (24 * 60) * 2 * math.pi for m in bedtimes_minutes]
        # Circular mean
        C = sum(math.cos(t) for t in rad) / len(rad)
        S = sum(math.sin(t) for t in rad) / len(rad)
        mean_rad = math.atan2(S, C)
        
        # Convert back to minutes (handling negative values)
        mean_minutes = (mean_rad / (2 * math.pi) * 24 * 60) % (24 * 60)
        
        avg_bedtime_hour = int(mean_minutes // 60)
        avg_bedtime_min = int(mean_minutes % 60)
        avg_bedtime = f"{avg_bedtime_hour:02d}:{avg_bedtime_min:02d}"
    
    # Calculate average wake time using circular mean (IN CET!)
    avg_wake_time = None
    if waketimes_minutes:
        # Convert to radians
        rad = [m / (24 * 60) * 2 * math.pi for m in waketimes_minutes]
        # Circular mean
        C = sum(math.cos(t) for t in rad) / len(rad)
        S = sum(math.sin(t) for t in rad) / len(rad)
        mean_rad = math.atan2(S, C)
        
        # Convert back to minutes (handling negative values)
        mean_minutes = (mean_rad / (2 * math.pi) * 24 * 60) % (24 * 60)
        
        avg_wake_hour = int(mean_minutes // 60)
        avg_wake_min = int(mean_minutes % 60)
        avg_wake_time = f"{avg_wake_hour:02d}:{avg_wake_min:02d}"
    
    # Time in bed (from first start to last end of each session)
    time_in_bed_total = 0
    for session in sleep_sessions:
        if session:
            first_start = session[0]["start"] if isinstance(session[0]["start"], datetime) else datetime.fromisoformat(str(session[0]["start"]).replace("Z", ""))
            last_end = session[-1]["end"] if isinstance(session[-1]["end"], datetime) else datetime.fromisoformat(str(session[-1]["end"]).replace("Z", ""))
            time_in_bed_total += (last_end - first_start).total_seconds() / 60
    
    return {
        "sleep_duration_min": round(total_sleep_min, 2),
        "sleep_duration_hours": round(total_sleep_min / 60, 2),
        "bed_activity_min": round(bed_activity_min, 2),
        "bed_activity_hours": round(bed_activity_min / 60, 2),
        "consistency_score": round(consistency_score, 2),
        "consistency_sd_minutes": round(consistency_sd, 2),
        "total_intervals": len(intervals),
        "avg_awakenings": round(avg_awakenings, 2),
        "wake_up_time": avg_wake_time,  # NOW IN CET!
        "bed_time": avg_bedtime,  # NOW IN CET!
        "time_in_bed_min": round(time_in_bed_total, 2),
        "time_in_bed_hours": round(time_in_bed_total / 60, 2),
        "nights_count": nights_count
    }



#calculating sleep metrics
# def compute_sleep_metrics(device_id: str) -> Dict:
#     """
#     Compute sleep metrics based on current occupancy and historical data.
#     Returns a dictionary with all metrics, even if errors occur.
#     """

#     now = datetime.now()
#     seven_days_ago = now - timedelta(days=7)
    
#     try:
#         response = supabase.table("raw_occupancy")\
#             .select("*")\
#             .eq("device_id", int(device_id))\
#             .gte("created_at", seven_days_ago.isoformat())\
#             .order("created_at")\
#             .execute()
        
#         occupancy_rows = response.data
#     except Exception as e:
#         print(f"Error fetching historical data: {e}")
#         return default_metrics
    
#     occ_intervals = build_occupancy_intervals(occupancy_rows)
    
#     if not occ_intervals:
#         return default_metrics
    
#     total_sleep_min = compute_total_sleep(occ_intervals)
#     total_nights = count_nights(occ_intervals)
#     avg_sleep = compute_avg_sleep_per_night(occ_intervals)
#     avg_awakenings = compute_avg_awakenings(occ_intervals)
    
#     bedtimes = []
#     for iv in occ_intervals:
#         bedtime_minutes = iv["start"].hour * 60 + iv["start"].minute
#         bedtimes.append(bedtime_minutes)
    
#     if bedtimes:
#         import statistics
#         consistency_sd = statistics.stdev(bedtimes) if len(bedtimes) > 1 else 0
#         consistency_score = max(0, 100 - (consistency_sd / 60 * 100))
#     else:
#         consistency_sd = 0
#         consistency_score = 0
    
#     bed_use_pct = (total_sleep_min / (7 * 24 * 60)) * 100 if total_sleep_min > 0 else 0
    
#     return {
#         "consistency_score": round(consistency_score, 2),
#         "consistency_sd_minutes": round(consistency_sd, 2),
#         "bed_on_coverage_pct": round(bed_use_pct, 2),
#         "avg_sleep_per_night_min": round(avg_sleep, 2),
#         "avg_sleep_per_night_hours": round((avg_sleep / 60), 2),
#         "total_intervals": len(occ_intervals),
#         "total_nights": total_nights,
#         "avg_awakenings": round(avg_awakenings, 2),
#         "total_sleep_min": round(total_sleep_min, 2),
#         "total_sleep_hours": round((total_sleep_min / 60), 2)
#     }

def build_occupancy_intervals(rows: List[Dict]) -> List[Dict]:
    """
    bools occupancy -> continuous intervals.
    """
    if not rows:
        return []
    
    samples = sorted(rows, key=lambda x: x["created_at"]) 
    MIN_SEG_SEC = 120 
    
    intervals = []
    in_occ = False
    current_start = None
    last_timestamp = None
    
    for sample in samples:
        timestamp = datetime.fromisoformat(sample["created_at"].replace("Z", "")) 
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
        "version": data.VersionFactory,
        "version_ota1": data.VersionOTA1,
        "version_ota2": data.VersionOTA2,
        "sd_free_storage": data.SDFreeStorage,
        "mac": data.MAC,
        "last_seen": datetime.now(timezone.utc).astimezone().isoformat()
    })
    
    try:
        device_record = {
            "id": int(device_id),  
            "name": data.CustomName,
            "status": data.Status,
            "temperature": data.Temperature,
            "mac": data.MAC,
            "last_seen": datetime.now(timezone.utc).astimezone().isoformat()
        }
        firmware_record = {
            "device_id": int(device_id),  
            "version": data.VersionFactory,
            "partition": data.ActualPartition,
            "version_ota1": data.VersionOTA1,
            "version_ota2": data.VersionOTA2,
            "sd_free_storage": data.SDFreeStorage,
            
        }
        device_metadata[device_id] = {
            "custom_name": data.CustomName,
        }
        
        result_devices = supabase.table("devices")\
            .upsert(device_record, on_conflict='id')\
            .execute()
        
        result_firmware = supabase.table("firmware")\
            .upsert(firmware_record, on_conflict='device_id')\
            .execute()
        
        print(f"upserted device {device_id}: {result_devices.data}")
        return {"success": True, "message": f"Device {device_id} registered", "device_id": device_id}
        
    except Exception as e:
        print(f"err upserting device {device_id}: {e}")
        return {"success": False, "message": f"Failed to register device: {str(e)}", "device_id": device_id}

    
@app.post("/esp32/{device_id}/sensors")
async def esp32_sensors(device_id: str, data: ESPSensorData):
    """ESP32 -> sends sensor readings"""
    global active_push_tasks
    
    try:
        result = supabase.table("devices").select("id").eq("id", int(device_id)).execute()
        print(f"Result from eq: {result}")
        if not result.data:
            # Device doesn't exist - log but don't fail
            print(f"⚠ Device {device_id} not found in database. Sensor data stored in memory only.")
            # Store in memory but skip database operations
            if device_id not in latest_data:
                latest_data[device_id] = {}
            
            latest_data[device_id].update({
                "occupancy": data.Occupancy,
                "speed": data.Speed,
                "acceleration": data.Acceleration,
                "vibration": data.Vibration,
                "intensity": data.Intensity,
                "safety": data.Safety,
                "motor_status": data.MotorStatus,
                "last_seen": datetime.now(timezone.utc).astimezone().isoformat()
            })
            
            #aggregate occ for when device gets registered
            aggregate_occupancy_binary(data.OccupancyBinary)
            
            return {
                "success": True, 
                "warning": "Device not registered. Call /startup endpoint first."
            }
        else:
            user_settings = {
                "device_id": int(device_id), 
                "vibration": data.Vibration,
                "intensity": data.Intensity,
                "motor_status": data.MotorStatus,
            }
            add_safety = {
                "id": int(device_id),
                "name": device_metadata[device_id]["custom_name"],
                "safety": data.Safety,
            }
            supabase.table("user_settings")\
                .upsert(user_settings, on_conflict='device_id')\
                .execute()
            supabase.table("devices")\
                .upsert(add_safety, on_conflict='id')\
                .execute()
    except Exception as e:
        print(f"Error checking device existence: {e}")
    
    # Update in-memory data
    if device_id not in latest_data:
        latest_data[device_id] = {}
    
    latest_data[device_id].update({
        "motor_status": data.MotorStatus,
        "occupancy": data.Occupancy,
        "speed": data.Speed,
        "acceleration": data.Acceleration,
        "vibration": data.Vibration,
        "intensity": data.Intensity,
        "safety": data.Safety,
        "last_seen": datetime.now(timezone.utc).astimezone().isoformat()
    })

    # Aggregate occupancy data
    aggregate_occupancy_binary(data.OccupancyBinary)
    
    # Start background task if not already running
    if device_id not in active_push_tasks or active_push_tasks[device_id].done():
        active_push_tasks[device_id] = asyncio.create_task(push_aggregated_data(device_id))
    
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
    try:
        response = supabase.table("devices")\
            .select("*")\
            .execute()
        
    except Exception as e:
        print(f"Error fetching devices: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        # return default_metrics
    
    # devices = []
    # for device_id, data in latest_data.items():
    #     devices.append({
    #         "device_id": device_id,
    #         **data
    #     })
    return response

#where the real-time values are fetched from specific ESP e.g. occupied
@app.get("/devices/{device_id}")
def get_device(device_id: str):
    """Get specific device data"""

    # if device_id not in latest_data:
    #     raise HTTPException(status_code=404, detail="Device not found")
    
    response = supabase.table("devices").select("*").eq("id", int(device_id)).execute()
    
    # echo
    return response

#User retrieves user settings
@app.get("/devices/{device_id}/settings")
def get_device(device_id: str):
    """Get device settings"""

    # if device_id not in latest_data:
    #     raise HTTPException(status_code=404, detail="Device not found")
    
    response = supabase.table("user_settings").select("*").eq("device_id", int(device_id)).execute()
    
    # echo
    return response

#Check if device is still online via occupied timestamp
@app.get("/devices/{device_id}/last_occupied")
def get_device(device_id: str):
    """Get most recent timestamp"""
    try:
        response = supabase.table("raw_occupancy")\
            .select("*")\
            .eq("device_id", int(device_id))\
            .order('created_at', desc=True)\
            .limit(1)\
            .execute()
        
        return response
    except Exception as e:
        print(f"Error fetching data: {e}")
    

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


@app.get("/user-settings/{device_id}")
async def get_user_settings(device_id: str):
    """Get user settings for a device"""
    try:
        print(f"Fetching user settings for device {device_id}")
        response = supabase.table("user_settings")\
            .select("*")\
            .eq("device_id", int(device_id))\
            .execute()
        
        print(f"User settings query result: {response.data}")
        return {"data": response.data, "count": len(response.data) if response.data else 0}
    except Exception as e:
        print(f"Error fetching user settings: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/sleep/{device_id}/summary")
async def get_sleep_summary(
    device_id: str,
    period: Literal["day", "week", "month"] = Query(default="week"),
    date: Optional[str] = Query(default=None, description="ISO date string (YYYY-MM-DD)"),
    sleep_boundary_hour: Optional[int] = Query(default=None, description="Hour of day where sleep day starts (0-23)")
):
    """
    Get sleep summary for a specific period (day/week/month)
    Now correctly handles sleep sessions that span across calendar days
    """
    try:
        print(f"\n=== Sleep Summary Request ===")
        print(f"Device: {device_id}, Period: {period}, Date: {date}, Boundary: {sleep_boundary_hour}")
        
        # Parse target date - make it timezone aware (UTC)
        if date:
            try:
                target_date = datetime.fromisoformat(date)
                if target_date.tzinfo is None:
                    target_date = target_date.replace(tzinfo=timezone.utc)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        else:
            target_date = datetime.now(timezone.utc)
        
        # Calculate date range based on period and sleep boundary
        if sleep_boundary_hour is not None:
            # Use sleep-centric day boundaries
            if period == "day":
                # The key fix: Dec 10 sleep day should capture sleep from previous evening
                # So we look BACK from the boundary, not forward
                # Dec 10 @ 03:00 means we want sleep from Dec 9 03:00 to Dec 10 03:00
                end_date = target_date.replace(hour=sleep_boundary_hour, minute=0, second=0, microsecond=0)
                start_date = end_date - timedelta(days=1)
                
                # Expand fetch range to ensure we capture everything
                fetch_start_date = start_date - timedelta(hours=12)
                fetch_end_date = end_date + timedelta(hours=12)
                
            elif period == "week":
                days_since_monday = target_date.weekday()
                week_start = target_date - timedelta(days=days_since_monday)
                end_date = week_start.replace(hour=sleep_boundary_hour, minute=0, second=0, microsecond=0) + timedelta(days=7)
                start_date = end_date - timedelta(days=7)
                
                fetch_start_date = start_date - timedelta(hours=12)
                fetch_end_date = end_date + timedelta(hours=12)
                
            elif period == "month":
                month_start = target_date.replace(day=1)
                if month_start.month == 12:
                    next_month = month_start.replace(year=month_start.year + 1, month=1)
                else:
                    next_month = month_start.replace(month=month_start.month + 1)
                
                end_date = next_month.replace(hour=sleep_boundary_hour, minute=0, second=0, microsecond=0)
                start_date = month_start.replace(hour=sleep_boundary_hour, minute=0, second=0, microsecond=0)
                
                fetch_start_date = start_date - timedelta(hours=12)
                fetch_end_date = end_date + timedelta(hours=12)
            else:
                raise HTTPException(status_code=400, detail="Invalid period")
        else:
            # Use standard calendar day boundaries (00:00-23:59)
            if period == "day":
                start_date = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = start_date + timedelta(days=1)
                fetch_start_date = start_date
                fetch_end_date = end_date
                
            elif period == "week":
                days_since_monday = target_date.weekday()
                week_start = target_date - timedelta(days=days_since_monday)
                start_date = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = start_date + timedelta(days=7)
                fetch_start_date = start_date
                fetch_end_date = end_date
                
            elif period == "month":
                start_date = target_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                if target_date.month == 12:
                    end_date = target_date.replace(year=target_date.year + 1, month=1, day=1)
                else:
                    end_date = target_date.replace(month=target_date.month + 1, day=1)
                fetch_start_date = start_date
                fetch_end_date = end_date
            else:
                raise HTTPException(status_code=400, detail="Invalid period")
        
        print(f"Fetch range: {fetch_start_date} to {fetch_end_date}")
        print(f"Logical range: {start_date} to {end_date}")
        
        # Fetch occupancy data from database
        response = supabase.table("raw_occupancy")\
            .select("*")\
            .eq("device_id", int(device_id))\
            .gte("created_at", fetch_start_date.isoformat())\
            .lte("created_at", fetch_end_date.isoformat())\
            .order("created_at")\
            .execute()
        
        occupancy_rows = response.data
        print(f"Fetched {len(occupancy_rows) if occupancy_rows else 0} occupancy rows")
        
        # Build intervals from occupancy data
        all_intervals = build_occupancy_intervals(occupancy_rows)
        print(f"Built {len(all_intervals)} total intervals")
        
        # NEW: Better interval filtering that considers sleep sessions spanning boundaries
        filtered_intervals = []
        for iv in all_intervals:
            interval_start = iv["start"] if isinstance(iv["start"], datetime) else datetime.fromisoformat(str(iv["start"]).replace("Z", "+00:00"))
            interval_end = iv["end"] if isinstance(iv["end"], datetime) else datetime.fromisoformat(str(iv["end"]).replace("Z", "+00:00"))
            
            # Ensure timezone awareness
            if interval_start.tzinfo is None:
                interval_start = interval_start.replace(tzinfo=timezone.utc)
            if interval_end.tzinfo is None:
                interval_end = interval_end.replace(tzinfo=timezone.utc)
            
            # Key change: Include interval if ANY part overlaps with range
            # OR if it's a sleep session that naturally belongs to this period
            # (starts before but ends during, or starts during)
            overlaps = interval_end >= start_date and interval_start < end_date
            
            if overlaps:
                filtered_intervals.append(iv)
                print(f"  ✓ Including interval: {interval_start} to {interval_end}")
            else:
                print(f"  ✗ Excluding interval: {interval_start} to {interval_end}")
        
        print(f"Filtered to {len(filtered_intervals)} intervals within range")
        
        # Compute metrics for this range
        metrics = compute_sleep_metrics_for_range(filtered_intervals, start_date, end_date)
        
        # Prepare intervals for response
        intervals_response = []
        for iv in filtered_intervals:
            intervals_response.append({
                "start": iv["start"].isoformat() if isinstance(iv["start"], datetime) else str(iv["start"]),
                "end": iv["end"].isoformat() if isinstance(iv["end"], datetime) else str(iv["end"]),
                "duration_min": iv["duration_min"]
            })
        
        result = {
            "device_id": device_id,
            "period": period,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "sleep_boundary_hour": sleep_boundary_hour,
            "summary": {
                **metrics,
                "intervals": intervals_response
            }
        }
        
        print(f"Returning summary with {len(intervals_response)} intervals")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR in get_sleep_summary: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") 

# @app.get("/sleep/{device_id}/summary")
# def get_sleep_summary(device_id: str):
#     """Get full sleep summary for dashboard"""

#     metrics: Dict = compute_sleep_metrics(device_id)
#     devices = supabase.table("devices").select("*").eq("id", int(device_id)).execute()

#     # debuggin
#     print(f"[DEBUG] Summary requested for device: {device_id}")
#     print(f"[DEBUG] Available device: {devices.data}")
#     # print(f"[DEBUG] Sleep data devices: {list(sleep_data.keys())}")

#     return {
#         "device_id": device_id,
#         "summary": {
#             "created_at": datetime.now().isoformat(),
#             "sleep_consistency": metrics.get("consistency_score", 0),
#             "bedtime_consistency": metrics.get("consistency_sd_minutes", 0),
#             "bed_use": metrics.get("bed_on_coverage_pct", 0),

#             # average values
#             "daily_occupancy": metrics.get("avg_sleep_per_night", 0), 
#             "avg_sleep_per_night_min": metrics.get("avg_sleep_per_night_min", 0),
#             "avg_sleep_per_night_hours": metrics.get("avg_sleep_per_night_hours", 0),
#             "avg_awakenings_per_night": metrics.get("avg_awakenings", 0),

#             # total values
#             "total_intervals": metrics.get("total_intervals", 0),
#             "total_nights": metrics.get("total_nights", 0),
#             "total_sleep_min": metrics.get("total_sleep_min", 0),
#             "total_sleep_hours": metrics.get("total_sleep_hours", 0),

#             "intervals": occ_intervals
#         }
#     }

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