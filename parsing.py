import os
import pandas as pd
from bs4 import BeautifulSoup

def extract_messages_from_html(file_path):
    # Open the file and parse it with BeautifulSoup
    with open(file_path, "r", encoding="windows-1251") as file:
        soup = BeautifulSoup(file, "html.parser")

    # Find all message blocks
    message_blocks = soup.find_all("div", class_="message")

    # Initialize an empty list to store the messages
    messages = []

    # Iterate over each message block and extract the data
    for message in message_blocks:
        # Extract the header
        header = message.find("div", class_="message__header")

        # Extract the user link if it exists
        if header.a is not None:
            user_link = header.a.get("href")
        else:
            user_link = "unknown"

        user_name, timestamp = header.get_text().rsplit(",", 1)

        # Extract the message text
        message_text = header.find_next_sibling("div").get_text(strip=True)

        # Store the data in a dictionary and add it to the list
        messages.append({
            "user_link": user_link,
            "user_name": user_name,
            "timestamp": timestamp.strip(),
            "message_text": message_text
        })

    return messages

# Specify the directory containing the HTML files
directory = "C:/Аня"

# Get a list of all HTML files in the directory
html_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".html")]

# Initialize an empty DataFrame to store all messages
df_all_messages = pd.DataFrame()

# Iterate over each HTML file and extract the messages
for html_file in html_files:
    messages = extract_messages_from_html(html_file)
    df_messages = pd.DataFrame(messages)
    df_all_messages = pd.concat([df_all_messages, df_messages], ignore_index=True)


# Save the DataFrame to a CSV file
df_all_messages.to_csv("C:/messages.csv", index=False)
