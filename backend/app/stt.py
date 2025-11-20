import os
try:
    import whisper
except Exception:
    whisper = None
from pathlib import Path
def get_model():
    global _MODEL
    try:
        _MODEL
    except NameError:
        _MODEL = None
    if _MODEL is None and whisper is not None:
        _MODEL = whisper.load_model('small')
    return _MODEL

def speech_to_text(path: str) -> str:
    if whisper is None:
        return "[whisper not installed] " + Path(path).name
    model = get_model()
    res = model.transcribe(path)
    return res.get('text', '').strip()
