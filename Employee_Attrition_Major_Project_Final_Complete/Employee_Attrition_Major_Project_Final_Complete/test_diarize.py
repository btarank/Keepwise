# test_diarize.py
from pyannote.audio import Pipeline
p = Pipeline.from_pretrained("pyannote/speaker-diarization")
print("Loaded pyannote pipeline:", p)
