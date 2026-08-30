# Packages
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import pandas as pd

# Config
login_time = 30                 # Time for login (in seconds)
new_msg_time = 5                # TTime for a new message (in seconds)
send_msg_time = 5               # Time for sending a message (in seconds)
country_code = 91               # Set your country code
action_time = 2                 # Set time for button click action
image_path = 'test.png'        # Absolute path to you image

data = pd.DataFrame({"contact":[9413608059,7400020639],"message":["Hello Gaurav","Hello ABC"]})

# Create driver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
# Open browser with default link
link = 'https://web.whatsapp.com'
driver.get(link)
time.sleep(login_time)

for index, row in data.iterrows():
    phone = str(row['contact'])
    message = row['message']
    #file_path = row['file']
    link = f'https://web.whatsapp.com/send/?phone={country_code}{phone}'
    driver.get(link)
    time.sleep(new_msg_time)
        # Click on button to load the input DOM
    if(image_path):
        attach_btn = driver.find_element(By.CSS_SELECTOR, '._1OT67')
        attach_btn.click()
        time.sleep(action_time)
        # Find and send image path to input
        msg_input = driver.find_elements(By.CSS_SELECTOR, '._2UNQo input')[1]
        msg_input.send_keys(image_path)
        time.sleep(action_time)
    # Start the action chain to write the message
    actions = ActionChains(driver)
    
    actions.send_keys(message)
            # SHIFT + ENTER to create next line
    actions.key_down(Keys.SHIFT).send_keys(Keys.ENTER).key_up(Keys.SHIFT)
    actions.send_keys(Keys.ENTER)
    actions.perform()
    time.sleep(send_msg_time)

# Quit the driver
driver.quit()