import platform
import threading
import time
import logging

# Check OS to handle the actual beeping
OS_NAME = platform.system()
if OS_NAME == "Windows":
    import winsound

def _play_tone(frequency, duration_ms):
    """Internal function to play a raw frequency."""
    if OS_NAME == "Windows":
        winsound.Beep(int(frequency), duration_ms)
    else:
        # Fallback for Linux/Mac. To make real tones on Linux, you'd usually 
        # use `os.system(f"play -n synth {duration_ms/1000} sine {frequency}")` 
        # requiring 'sox' to be installed.
        logging.info(f"[Audio Alert] Beep! Freq: {frequency}Hz for {duration_ms}ms")
        time.sleep(duration_ms / 1000.0)

def play_in_background(fn, *args, **kwargs):
    """Run a ringtone function in a background daemon thread (non-blocking).

    Returns the started :class:`threading.Thread` so callers can optionally
    ``join()`` it.  Using a *daemon* thread means the audio will not block the
    process from exiting if it finishes before the sound completes.
    """
    t = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
    t.start()
    return t


def alert_success(enabled=True):
    """Triumphant arpeggio for script completion."""
    if not enabled: return
    notes = [(261.63, 150), (329.63, 150), (392.00, 150), (523.25, 400)]
    for freq, duration in notes:
        if freq > 0:  
            _play_tone(freq, duration)
        else:
            time.sleep(duration / 1000.0)
        time.sleep(0.1) # Short pause between notes

def alert_finish(enabled=True):
    """
    Spielt eine feierliche Abschluss-Fanfare:
    """
    if not enabled: 
        return

    notes = [
        (523.25, 200),
        (659.25, 200),
        (783.99, 200),
        (1046.50, 400),
        (987.77, 200),
        (783.99, 200),
        (659.25, 200),
        (587.33, 200),
        (783.99, 200),
        (987.77, 200),
        (1318.51, 200),
        (1046.50, 400),
        (0, 600), 
        (783.99, 300),
        (1046.50, 700),
        (783.99 , 200),
        (1046.50, 900),
        (783.99 , 200),
        (1046.50, 900),
        (783.99, 250),
        (659.25, 250),
        (523.25, 700)
    ]
    for freq, duration in notes:
        if freq > 0:
            _play_tone(freq, duration)
        else:
            time.sleep(duration / 1000.0)
        time.sleep(0.05)

def alert_failure(enabled=True):
    """Sad trombone for when things go terribly wrong."""
    if not enabled: return
    notes = [(622.25, 300), (587.33, 300), (554.37, 300), (523.25, 800)]
    for freq, duration in notes:
        if freq > 0:  
            _play_tone(freq, duration)
        else:
            time.sleep(duration / 1000.0)
    time.sleep(0.2) # Tiny pause for dramatic, sad effect

def alert_advancement(enabled=True):
    """Quick ding-ding for checkpoints."""
    if not enabled: return
    notes = [(783.99, 75), (1046.50, 150), (0, 500), (783.99, 75), (1046.50, 150)]
    for freq, duration in notes:
        if freq > 0:  
            _play_tone(freq, duration)
        else:
            time.sleep(duration / 1000.0)
        time.sleep(0.075) # Short pause to let the ding-ding be distinct

def alert_warning(enabled=True):
    """Siren oscillation for warnings or resource limits."""
    if not enabled: return
    # Alternating high/low dissonant notes
    notes = [
        (1046.50, 400), (739.99, 400), 
        (1046.50, 400), (739.99, 400)
    ]
    for freq, duration in notes:
        if freq > 0:
            _play_tone(freq, duration)
        else:
            time.sleep(duration / 1000.0)
        time.sleep(0.05) # Tiny gap


def alert_input_required(enabled=True):
    """Urgent, repetitive beeps prompting user action."""
    if not enabled: return
    # Three fast beeps, a pause, three fast beeps
    notes = [
        (987.77, 100), (0, 100), (987.77, 100), (0, 100), (987.77, 100),
        (0, 1200), # Longer pause before repeating
        (987.77, 100), (0, 100), (987.77, 100), (0, 100), (987.77, 100)
    ]
    for freq, duration in notes:
        if freq > 0:
            _play_tone(freq, duration)
        else:
            time.sleep(duration / 1000.0)

def alert_heartbeat(enabled=True):
    """A low, unobtrusive double-thump to indicate active processing."""
    if not enabled: return
    # No print statement here so it doesn't spam your logs!
    notes = [
        (261.63, 150), # Thump
        (0, 100),      # Small pause to simulate a heartbeat rhythm
        (261.63, 150),  # Thump
        (0, 500),      # Longer pause to simulate a heartbeat rhythm
        (261.63, 150), # Thump
        (0, 100),      # Small pause to simulate a heartbeat rhythm
        (261.63, 150)  # Thump
    ]
    for freq, duration in notes:
        if freq > 0:
            _play_tone(freq, duration)
        else:
            time.sleep(duration / 1000.0)


import contextlib
import sys

@contextlib.contextmanager
def monitor_process(enabled=True):
    """
    A context manager that automatically plays success/failure tones
    based on whether the enclosed code completes or raises an exception.
    """
    try:
        yield # Run the code inside the 'with' block
        # If we get here, the code succeeded!
        alert_success(enabled=enabled)
    except Exception as e:
        # If we get here, something crashed!
        print(f"\n[!] ERROR DETECTED: {e}", file=sys.stderr)
        alert_failure(enabled=enabled)
        raise  # Re-raise preserving the original traceback
    
import logging

class AudioAlertHandler(logging.Handler):
    """
    Custom logging handler that plays specific ringtones based on log severity.

    A 2-second cooldown between alerts prevents an unbounded number of background
    threads when a noisy dependency emits repeated warnings.
    """
    _COOLDOWN_SECONDS = 2.0

    def __init__(self, enabled=True):
        super().__init__()
        self.enabled = enabled
        self._last_played: float = 0.0

    def emit(self, record):
        # If audio is off, do nothing
        if not self.enabled:
            return

        now = time.time()
        if now - self._last_played < self._COOLDOWN_SECONDS:
            return  # Rate-limit: skip if a tone was played very recently
        self._last_played = now

        # Map log levels to our symphony tracks — played in the background so
        # that logging calls never block the main thread / progress bar.
        if record.levelno >= logging.CRITICAL:
            play_in_background(alert_failure, enabled=True)
        elif record.levelno >= logging.ERROR:
            play_in_background(alert_failure, enabled=True)
        elif record.levelno >= logging.WARNING:
            play_in_background(alert_warning, enabled=True)
        # We generally skip INFO and DEBUG so your script doesn't sound
        # like a 1990s arcade on turbo mode.


def setup_audio_logger(enabled=True, level=logging.INFO):
    """
    Attaches an :class:`AudioAlertHandler` to the root logger.

    Unlike the previous implementation this function does **not** clear any
    existing handlers, so calling it after ``logging.basicConfig`` (as both
    CLI scripts do) preserves the already-configured console formatter.
    """
    logger = logging.getLogger()
    if level is not None:
        logger.setLevel(min(logger.level or logging.WARNING, level))

    # Only add the audio handler if one is not already attached, so that
    # calling setup_audio_logger() twice doesn't double-play every tone.
    if not any(isinstance(h, AudioAlertHandler) for h in logger.handlers):
        audio_handler = AudioAlertHandler(enabled=enabled)
        logger.addHandler(audio_handler)

    return logger