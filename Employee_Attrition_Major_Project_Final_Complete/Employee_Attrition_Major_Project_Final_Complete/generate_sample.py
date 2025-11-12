from gtts import gTTS

# Define a short meeting dialogue
meeting_text = """
Manager: Hi everyone, let's discuss how work has been this month.
Employee 1: I am a bit tired and feel overworked lately.
Employee 2: I'm happy with my tasks and the new opportunities.
Employee 3: I'm thinking of finding another job soon if things don't improve.
"""

# Create TTS audio
tts = gTTS(meeting_text, lang="en")

# Save as mp3
tts.save("sample_meeting.mp3")
print("✅ Sample meeting audio saved as 'sample_meeting.mp3'")
