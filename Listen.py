import fnmatch
import os


def find_executable(app_name):
    # Common installation directories
    search_paths = [
        "C:\\Program Files",
        "C:\\Program Files (x86)",
        os.path.expanduser("~\\AppData\\Local")
    ]

    # Patterns to match executable files
    patterns = [f"{app_name}.exe"]

    # Recursively search for executable files
    for search_path in search_paths:
        for root, dirs, files in os.walk(search_path):
            for pattern in patterns:
                for filename in fnmatch.filter(files, pattern):
                    return os.path.join(root, filename)

    return None


app_path = find_executable("WhatsApp")
print(app_path)
# os.startfile(app_path)
