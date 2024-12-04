from streampot import StreamPot
import os

client = StreamPot(secret=os.getenv("STREAMPOT_API_TOKEN"))

job = client.input("test_t.wav") \
    .output("test_t.mp3") \
    .run_and_wait()

print(job)
