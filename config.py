# config.py
import os
from dataclasses import dataclass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class Config:
    BASE_DIR: str = BASE_DIR
    # Camera
    CAMERA_URL: str = "http://10.84.152.163:8080/video"
    ROTATE: int = 0  # 0, 90, 180, 270

    # ROI auto-crop (remove big white/blank area)
    ROI_AUTO_CROP: bool = True
    ROI_WHITE_THRESH: int = 245
    ROI_MIN_CONTENT_RATIO: float = 0.15
    ROI_PAD: int = 10

    # Person model (detection + tracking)
    YOLO_MODEL: str = "yolov8n-pose.pt"  # start with n, later you can use yolov8m-pose.pt
    DEVICE: str = "0"              # "0" for GPU, "cpu" for CPU
    CONF: float = 0.30
    IMGSZ: int = 640

    # Boundary
    BOUNDARY_FILE: str = os.path.join(BASE_DIR, "boundary_points.json")
    NEAR_DISTANCE_PX: float = 70.0
    EDGE_EPS_PX: float = 25.0

    # Timer intrusion
    INTRUSION_SECONDS: float = 2.0

    # Evidence
    SAVE_DIR: str = os.path.join(BASE_DIR, "ai_catcher_evidence")
    EVENTS_FILE: str = os.path.join(SAVE_DIR, "events.jsonl")
    WARNING_IMG_COOLDOWN_SEC: float = 3.0
    INTRUSION_CLIP_PRE_SEC: float = 3.0
    INTRUSION_CLIP_POST_SEC: float = 20.0
    RECORD_FPS: int = 20

    # Dashboard
    DASHBOARD_HOST: str = "127.0.0.1"
    DASHBOARD_PORT: int = 3000
    STATUS_HOLD_SEC: float = 2.5

    # Layer-4 Violence (Heuristic: Prox + Speed)
    HEURISTIC_INTERACT_DISTANCE_PX: float = 350.0
    HEURISTIC_HIGH_SPEED_PX_PER_SEC: float = 50.0  # further lowered for low-fps feeds
    
    FIGHT_CONF: float = 0.40 # lowered to match relaxed pipeline
    FIGHT_CHECK_EVERY_N_FRAMES: int = 1 # Check every frame so nothing is missed
    FIGHT_HITS_TO_CONFIRM: int = 2 # Requires 2 frames of high confidence
    FIGHT_HIT_DECAY_SEC: float = 2.0

    # Sounds
    INTRUSION_BEEP_ENABLED: bool = True
    CRITICAL_SIREN_ENABLED: bool = True
    CRITICAL_SIREN_WAV: str = os.path.join(BASE_DIR, "alarm", "siren1.wav")

    # Dynamic Dashboard Overrides
    INTRUSION_SENSITIVITY_PCT: int = 70
    VIOLENCE_SENSITIVITY_PCT: int = 60
    FIGHT_MODULE_ENABLED: bool = True
    FIGHT_SENSITIVITY_PCT: int = 75

    def save_dynamic(self):
        import json
        path = os.path.join(BASE_DIR, "dynamic_settings.json")
        data = {
            "INTRUSION_BEEP_ENABLED": self.INTRUSION_BEEP_ENABLED,
            "CRITICAL_SIREN_ENABLED": self.CRITICAL_SIREN_ENABLED,
            "INTRUSION_SENSITIVITY_PCT": self.INTRUSION_SENSITIVITY_PCT,
            "VIOLENCE_SENSITIVITY_PCT": self.VIOLENCE_SENSITIVITY_PCT,
            "FIGHT_MODULE_ENABLED": self.FIGHT_MODULE_ENABLED,
            "FIGHT_SENSITIVITY_PCT": self.FIGHT_SENSITIVITY_PCT,
            "INTRUSION_CLIP_PRE_SEC": self.INTRUSION_CLIP_PRE_SEC,
            "INTRUSION_CLIP_POST_SEC": self.INTRUSION_CLIP_POST_SEC
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load_dynamic(self):
        import json
        path = os.path.join(BASE_DIR, "dynamic_settings.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                for k, v in data.items():
                    if hasattr(self, k):
                        setattr(self, k, v)
            except Exception as e:
                pass


    # Critical evidence + cooldown
    THREAT_CLIP_PRE_SEC: float = 4.0
    THREAT_CLIP_POST_SEC: float = 20.0
    CRITICAL_COOLDOWN_SEC: float = 5.0

    # Branding
    APP_NAME: str = "AI Catcher"
    VERSION: str = "v1.0"

    # Multiple admins (prototype)
    DASHBOARD_ADMINS = [
    {"username": "admin", "password": "admin123"},
    {"username": "kugan", "password": "kugan123"},
    {"username": "security", "password": "sec123"},
]

DASHBOARD_SECRET_KEY: str = "ai_catcher_secret_change_me"
cfg = Config()