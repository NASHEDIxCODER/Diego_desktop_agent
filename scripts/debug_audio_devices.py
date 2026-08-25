#!/usr/bin/env python3
import sounddevice as sd
import traceback

def try_open(device, name):
    print(f"\n=== Testing device: {device} -> {name}")
    try:
        print('Hostapis:', sd.query_hostapis())
    except Exception as e:
        print('Hostapi query failed:', e)
    try:
        print('Default device:', sd.default.device)
    except Exception as e:
        print('Default device query failed:', e)
    try:
        dev = sd.query_devices(device)
        print('Device info:', dev)
    except Exception as e:
        print('query_devices failed:', e)
    for channels in (2,1):
        try:
            print(f"Trying InputStream channels={channels}")
            stream = sd.InputStream(device=device, channels=channels, dtype='float32')
            stream.start()
            stream.stop()
            stream.close()
            print(f"InputStream opened with channels={channels} OK")
        except Exception as e:
            print(f"InputStream failed channels={channels}: {e}")
    try:
        print('Trying rec 0.5s, channels=1')
        rec = sd.rec(int(0.5 * int(dev.get('default_samplerate') or 44100)), samplerate=int(dev.get('default_samplerate') or 44100), channels=1, dtype='int16', device=device, blocking=True)
        print('rec ok shape', getattr(rec,'shape',None))
    except Exception as e:
        print('rec failed:', e)


def main():
    print('All devices:')
    try:
        for i,d in enumerate(sd.query_devices()):
            print(i, d)
    except Exception as e:
        print('query_devices failed overall:', e)

    # Replace with indices seen in logs
    for idx in (5,7,8,9):
        try:
            info = sd.query_devices(idx)
            name = info.get('name') if isinstance(info, dict) else str(info)
        except Exception:
            name = '<query failed>'
        try:
            try_open(idx, name)
        except Exception:
            traceback.print_exc()

if __name__ == '__main__':
    main()
