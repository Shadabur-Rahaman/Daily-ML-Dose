# event_trigger.py
import time
from retrain_pipeline import retrain_model

def listen_for_events():
    while True:
        if check_drift_detected():  # custom function
            retrain_model()
        time.sleep(3600)

if __name__ == "__main__":
    listen_for_events()
