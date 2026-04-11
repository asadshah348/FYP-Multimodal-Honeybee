import pandas as pd
import webbrowser
import time
import pyautogui
import urllib.parse

# Excel file path
file = r"C:\Users\DELL\Downloads\contactss" \
".xlsx"

# Load Excel
data = pd.read_excel(file)
data.columns = data.columns.str.strip()

message = """GikiMovers – EID Break Bookings Are Live! 🌙✨

السلام علیکم !

EID Break bookings are now LIVE for GikiMovers.

👉 Book now using the form & details are mentioned in it:
https://docs.google.com/forms/d/e/1FAIpQLSf75sDmTDA3slp2ZfLps7bMlTLu0uCdnRt8SdUvuB-DgbwnUg/viewform

Book early and travel stress-free this Eid with GikiMovers. 🚌💙
"""

encoded_message = urllib.parse.quote(message)

for index, row in data.iterrows():

    phone = str(row['Contact no.']).strip()

    if not phone.startswith("+"):
        phone = "+92" + phone[-10:]

    print("Sending message to:", phone)

    url = f"https://web.whatsapp.com/send?phone={phone}&text={encoded_message}"

    webbrowser.open(url)

    # wait for WhatsApp to load
    time.sleep(15)

    # press enter to send message
    pyautogui.press("enter")

    # wait before next message
    time.sleep(10)

print("✅ All messages processed.")