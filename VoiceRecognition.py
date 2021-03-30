import queue
import sounddevice as sd
import vosk
import sys

q = queue.Queue()


def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))


try:
    device_info = sd.query_devices(0, 'input')
    samplerate = int(device_info['default_samplerate'])

    model = vosk.Model("model")

    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=0, dtype='int16', channels=1, callback=callback):

        rec = vosk.KaldiRecognizer(model, samplerate)
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                    result = rec.Result()
                    result = result[result.find('"text"'):-1]
                    result = result[result.find('"',7)+1:-2]
                    print("you said: " + result)

except KeyboardInterrupt:
    print('\nDone')
