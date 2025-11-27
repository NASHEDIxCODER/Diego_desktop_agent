import subprocess

def set_brightness(value: int):
    """
    Set brightness to specific percentage (0-100).
    """

    if not isinstance(value, int):
        raise ValueError("Brightness must be an integer")

    if not 0 <= value <= 100:
        raise ValueError("Brightness must be between 0 and 100")

    try:
        result = subprocess.run(
            ["brightnessctl", "s", f"{value}%"],
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print("Brightness command failed:", e.stderr.strip())
        return False
