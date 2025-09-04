from bs4 import BeautifulSoup
import requests
import time

script = input("enter stock symbol: ").upper()
url = f'https://www.google.com/finance/quote/{script}:NSE'

for i in range(3):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    price = float(soup.find(class_='YMlKec fxKbKc').text.strip()[1:].replace(",",""))
    print(price)
    time.sleep(10)