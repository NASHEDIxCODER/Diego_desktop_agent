import os
from datetime import datetime

import google.generativeai as genai

# Configure the API key using the environment variable
genai.configure(api_key=os.environ["GoogleApikey"])

# Proceed with using the GenerativeAI library
model = genai.GenerativeModel('gemini-1.0-pro-latest')
current_time = datetime.now()
time_str = current_time.strftime("%Y-%m-%d__%H-%M-%S")

log_dir = ('gem_guery')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def gemini(query):
    response = model.generate_content(query)
    print(response.text)
    replaced_response = response.text.replace("*", " ")
    print(replaced_response)
    # for logs__________
    log_file_path = os.path.join(log_dir, f'log_{time_str}.txt')
    with open(log_file_path, 'w') as log_file:
        log_file.write(f'log entry at {time_str}\n')
        log_file.write(f'The QUERY:>>>>>>>\n {query}\n')
        log_file.write(f'The RESPONSE:>>>>>>>\n{response.text}')

    return response.text


if __name__ == '__main__':
    q = input("enter your query: ")
    gemini(q)
