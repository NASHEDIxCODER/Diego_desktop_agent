import time

import pyautogui
from pytube import YouTube
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys



options = webdriver.ChromeOptions()
options.add_experimental_option("detach", True)
driver = webdriver.Chrome(options=options)

wait = WebDriverWait(driver, 15)


def youtube():
    driver.get("https://www.youtube.com/")
    # Wait for the search box to appear
    wait.until(EC.presence_of_element_located((By.NAME, "search_query")))


def search_song(n: str):
    # Make sure we're on YouTube
    if "youtube.com" not in driver.current_url:
        driver.get("https://www.youtube.com/")

    # Wait for the search box to be ready
    search = wait.until(
        EC.element_to_be_clickable((By.NAME, "search_query"))
    )
    search.clear()
    search.send_keys(n)
    search.send_keys(Keys.RETURN)  # ⬅️ no more search button needed

    # Wait for the first video result and click it
    first_result = wait.until(
        EC.element_to_be_clickable(
            (By.XPATH, "(//a[@id='video-title' and @href])[1]")
        )
    )
    first_result.click()
    print(f"Playing first result for: {n}")


def skip_ad():
    """
    Try to skip YouTube ad if skippable.
    """
    # Give time for ad to load if present
    time.sleep(5)
    try:
        # Newer layouts often use this structure:
        # button.ytp-ad-skip-button, or include "ytp-ad-skip-button" in class
        skip_button = wait.until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "//button[contains(@class,'ytp-ad-skip-button')]",
                )
            )
        )
        skip_button.click()
        print("Ad skipped")
    except Exception as e:
        # Either no ad, or non-skippable ad
        print(f"Ad not skippable or no ad: {e}")
        time.sleep(2)


def open_history():
    try:
        driver.get("https://www.youtube.com/feed/history")
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "ytd-item-section-renderer")))
        print("Opened YouTube history.")
    except Exception as e:
        print(f"Error opening history: {e}")


def play_next_song():
    try:
        next_button = wait.until(
            EC.element_to_be_clickable((By.CLASS_NAME, "ytp-next-button"))
        )
        next_button.click()
        print("Playing next song.")
    except Exception as e:
        print(f"Error playing next song: {e}")


def stop_song():
    # YouTube toggle play/pause is "k" or space
    pyautogui.press("k")
    print("Song has been stopped (toggled play/pause).")


def play_back_speed_i():
    # YouTube: increase playback speed = SHIFT + .
    pyautogui.hotkey("shift", ".")
    print("Increased playback speed.")


def play_back_speed_d():
    # YouTube: decrease playback speed = SHIFT + ,
    pyautogui.hotkey("shift", ",")
    print("Decreased playback speed.")


def download_song():
    current_url = driver.current_url
    print("Current URL:", current_url)

    try:
        yt = YouTube(current_url)
    except Exception as e:
        print(f"Error initializing YouTube object: {e}")
        return

    try:
        title = yt.title
        safe_title = "".join(x for x in title if x.isalnum() or x in "._- ")

        # Try progressive 480p stream (video+audio)
        stream = yt.streams.filter(progressive=True, res="480p").first()
        if not stream:
            # fallback: take first progressive stream
            stream = yt.streams.filter(progressive=True).first()

        if stream:
            stream.download(output_path="", filename=f"{safe_title}.mp4")
            print(f"Downloaded: {safe_title}.mp4")
        else:
            print("No suitable progressive stream found.")
    except Exception as e:
        print(f"Error downloading video: {e}")


if __name__ == "__main__":
    youtube()
    s = "blue eyes"
    search_song(s)
    skip_ad()
    # time.sleep(10)
    # stop_song()
    # play_next_song()
    # open_history()
    # download_song()
