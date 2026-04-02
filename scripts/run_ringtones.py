import argparse
import time
import logging
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src import ringtones

def main():
    parser = argparse.ArgumentParser(description="Runner Script with Audio Notifications")
    parser.add_argument('--make-noise', action='store_true', 
                        help='Enable audible ringtones for script events')
    args = parser.parse_args()

    AUDIO_ON = args.make_noise

    # Initialize our logger with the audio handler attached
    logger = ringtones.setup_audio_logger(enabled=AUDIO_ON)

    logger.info("Starting the arduous task...")
    
    with ringtones.monitor_process(enabled=AUDIO_ON):
        
        # Simulate Phase 1
        time.sleep(1) 
        
        # We manually call advancement here, since we didn't tie it to INFO logs
        ringtones.alert_advancement(enabled=AUDIO_ON)
        logger.info("Phase 1 complete. Moving to Phase 2.")
        
        # Simulate high memory or a weird data anomaly
        time.sleep(1)
        logger.warning("Memory usage spiking to 85%! Initiating garbage collection.")
        # ^ This line will AUTOMATICALLY play the dissonant "Red Alert" siren!
        
        time.sleep(1)
        
        # Simulate an actual failure based on a flag
        simulate_failure = False 
        if simulate_failure:
            # This logs the error, which triggers the AudioHandler (sad trombone),
            # AND the context manager will catch the raised exception.
            logger.error("The database refused the connection.")
            raise ConnectionError("Database timeout.")

        logger.info("Task completed successfully!")
        # Context manager automatically plays the Success tone here!

if __name__ == "__main__":
    main()