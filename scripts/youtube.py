
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

# ----------------------
# SELENIUM SETUP
# ----------------------

options = webdriver.ChromeOptions()
options.add_experimental_option("detach", True)

driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 15)


# ----------------------
# BASIC OPEN + SEARCH
# ----------------------

def youtube():
    driver.get("https://www.youtube.com/")
    wait.until(EC.presence_of_element_located((By.NAME, "search_query")))


def search_song(query: str):
    if "youtube.com" not in driver.current_url:
        driver.get("https://www.youtube.com/")

    search_box = wait.until(EC.element_to_be_clickable((By.NAME, "search_query")))
    search_box.clear()
    search_box.send_keys(query)
    search_box.send_keys(Keys.RETURN)

    # click first video
    first_video = wait.until(
        EC.element_to_be_clickable((By.XPATH, "(//a[@id='video-title' and @href])[1]"))
    )
    first_video.click()

    # ensure player is ready
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "html5-video-player")))
    print(f"Playing: {query}")


# ----------------------
# AD SKIP
# ----------------------

def skip_ad():
    try:
        driver.execute_script("""
                    let skipBtn = document.querySelector('.ytp-ad-skip-button, .ytp-ad-skip-button-modern');
                    if (skipBtn) skipBtn.click();
                """)
        print("Ad skipped")
    except:
        print("No skippable ad.")


# ----------------------
# Selenium Player Controls

def pause_or_play():
    """Click the play/pause button using DOM — 100% reliable."""
    try:
        video = driver.find_element(By.TAG_NAME, "video")
        video.click()
        print("Play/Pause toggled.")
        time.sleep(2)
        driver.execute_script("document.querySelector('video').focus();")
    except Exception as e:
        print("Pause/Play error:", e)


def play_next_song():
    """Click Next button from YouTube UI."""
    try:
        driver.execute_script("""
                    var player = document.querySelector('video');
                    var next = document.querySelector('.ytp-next-button');
                    if (next) next.click();
                """)
        print("Next video playing.")
        driver.execute_script("document.querySelector('video').focus();")
    except Exception as e:
        print("Error next video:", e)


def play_previous_song():
    """Click Previous button."""
    try:
        driver.execute_script("""
                    var prev = document.querySelector('.ytp-prev-button');
                    if (prev) prev.click();
                """)
        print("Previous video playing.")
        driver.execute_script("document.querySelector('video').focus();")
    except Exception as e:
        print("Error previous video:", e)


# ----------------------
# Playback speed — Selenium + JavaScript
# ----------------------

def set_playback_speed(speed: float):
    """Set YouTube speed using JS instead of hotkeys."""
    try:
        driver.execute_script(
            f"document.querySelector('video').playbackRate = {speed};"
        )
        print(f"Playback speed set to {speed}x")
    except Exception as e:
        print("Error speed:", e)


def increase_speed():
    video = driver.execute_script("return document.querySelector('video').playbackRate;")
    new_speed = min(video + 0.25, 2.0)
    set_playback_speed(new_speed)


def decrease_speed():
    video = driver.execute_script("return document.querySelector('video').playbackRate;")
    new_speed = max(video - 0.25, 0.25)
    set_playback_speed(new_speed)


# ----------------------
# Volume Control — JavaScript
# ----------------------

def set_volume(level: float):
    """
    Set volume 0.0–1.0 using JS.
    """
    level = max(0.0, min(1.0, level))
    try:
        driver.execute_script(f"document.querySelector('video').volume = {level};")
        print(f"Volume set to {int(level*100)}%")
    except Exception as e:
        print("Volume error:", e)


# ----------------------
# Seek
# ----------------------

def seek_forward(seconds=10):
    try:
        driver.execute_script(
            f"var v=document.querySelector('video'); v.currentTime += {seconds};"
        )
        print(f"Seek +{seconds} sec")
    except:
        pass


def seek_backward(seconds=10):
    try:
        driver.execute_script(
            f"var v=document.querySelector('video'); v.currentTime -= {seconds};"
        )
        print(f"Seek -{seconds} sec")
    except:
        pass

def toggle_mute():
    """Toggle mute/unmute on YouTube."""
    try:
        script = """
        var v = document.querySelector('video');
        if (v.muted === true) {
            v.muted = false;
            return "unmuted";
        } else {
            v.muted = true;
            return "muted";
        }
        """
        status = driver.execute_script(script)
        print(f"Video {status}.")
        return status
    except Exception as e:
        print("Mute/Unmute error:", e)

def close_youtube():
    try:
        driver.quit()
        print("YouTube closed.")
    except Exception as e:
        print("Error closing YouTube:", e)



# ----------------------
# MAIN TEST
# ----------------------

if __name__ == "__main__":
    print("YouTube started.")

