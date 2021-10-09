from sys import platform
import requests
from bs4 import BeautifulSoup
from colorama import Fore, Back, Style
import Dictionary

def CheckingOnWebsite(value):
    try:
        def CarModelAndColorInfo(a, b):
            print((soup.find_all('div', class_='')[a]).getText())
            print('Колір: ' + (soup.find_all('div', class_='')
                  [b].get_text()).replace('Цвет', ''))

        # url = "https://carsua.net/car-check/" + ReplaceInputTextCyrToLat(value)
        url = "https://carsua.net/car-check/" + Dictionary.ReplaceInputTextCyrToLat(value)

        html = requests.get(url).text
        soup = BeautifulSoup(html, 'lxml')
        div = soup.find_all('div', class_='text-gray-600')[2]
        if (div.get_text() == "Дата угона"):
            print(Fore.RED + "АВТО ВИКРАЛИ")
            CarModelAndColorInfo(9, 10)
            print(Style.RESET_ALL)
        elif (div.get_text() == "Авто"):
            print(Fore.GREEN + "Авто не в угоні")
            CarModelAndColorInfo(7, 8)
            print(Style.RESET_ALL)

    except IndexError:
        print(Fore.YELLOW + "Авто немає в базі")
        print(Style.RESET_ALL)
a = input()
CheckingOnWebsite(a)