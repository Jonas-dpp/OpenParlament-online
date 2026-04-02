import time
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src import ringtones

def run_demo():
    print("Welcome to the Terminal Symphony Orchestra. 🎻")
    print("Grab your popcorn. Playing the demo tracklist...\n")

    # Track 1
    print("\nTrack 1: 'The Checkpoint Ding' (Advancement)")
    ringtones.alert_advancement(enabled=True)
    time.sleep(1.5) # Pause to let the notes breathe

    # Track 2
    print("\nTrack 2: 'The Level Up' (Success)")
    ringtones.alert_success(enabled=True)
    time.sleep(1.5)

    # Track 3
    print("\nTrack 3: 'The Sad Trombone' (Failure)")
    ringtones.alert_failure(enabled=True)
    time.sleep(1.5)

    # Track 4
    print("\nTrack 4: 'The Red Alert' (Warning/High Memory)")
    ringtones.alert_warning(enabled=True)
    time.sleep(1.5)

    # Track 5
    print("\nTrack 5: 'Wake Up, Human' (Input Required)")
    ringtones.alert_input_required(enabled=True)
    time.sleep(1.5)
    # Track 6
    print("\nTrack 6: 'I'm Still Alive' (Heartbeat)")
    ringtones.alert_heartbeat(enabled=True)
    time.sleep(1.5)

    # Track 7
    print("\nTrack 7: 'The Finish Fanfare' (Finish)")
    ringtones.alert_finish(enabled=True)
    time.sleep(1.5) # Pause to let the notes breathe

    print("\nDemo complete! Your console is officially ready to serenade you.")

if __name__ == "__main__":
    run_demo()