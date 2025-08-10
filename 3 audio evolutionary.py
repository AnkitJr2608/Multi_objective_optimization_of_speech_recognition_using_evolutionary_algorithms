import sounddevice as sd
import soundfile as sf
import os
from datetime import datetime

# Set output directory (Windows Documents folder)
output_dir = os.path.expanduser("~/Documents")
sample_rate = 44100
duration = 4  # seconds

for i in range(1, 4):  # Will create 3 recordings
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"recording_{i}_{timestamp}.wav")
    
    # Record audio
    print(f"Recording {i}... (Speak now for {duration} seconds)")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording finishes
    
    # Process and save
    audio = audio.squeeze().astype('float32')  # Ensure correct format
    sf.write(output_file, audio, sample_rate, subtype='PCM_16')
    print(f"Saved: {output_file}\n")

print("All 3 recordings saved successfully!")