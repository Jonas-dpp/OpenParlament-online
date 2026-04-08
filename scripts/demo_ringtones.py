"""
OpenParlament - Streamlit Dashboard
Copyright (C) 2026 Jonas-dpp

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
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