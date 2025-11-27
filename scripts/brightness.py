import subprocess

def set_brightness(value):
    """Set brightness to specific percentage (0-100)."""
    try:
        subprocess.run(["brightnessctl", "s", f"{value}%"], check=False)
    except Exception as e:
        print("Brightness error:", e)

