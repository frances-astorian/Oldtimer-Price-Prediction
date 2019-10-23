import re
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver


def scrape_oldtimer_links(url="https://bringatrailer.com/auctions/results/"):
    """Creates a list of Links containing recent auction results from BringATrailer

    Args:
        url: (str) The url from where to scrape auction result links

    Returns:
        A list containing links to recent auctions
    """

    driver = webdriver.Chrome()
    driver.implicitly_wait(2)
    car_page_links_df = pd.read_csv('auction_result_links.csv', index_col=0, names=['url'])
    car_page_links = set(car_page_links_df.url)

    for page in range(100):
        print('Page : ', page)
        try:
            view_more_button = driver.find_element_by_class_name("button.auctions-footer-button")
            view_more_button.click()
            driver.implicitly_wait(6)

            if page % 10 == 0:
                page_source = driver.page_source
                soup = BeautifulSoup(page_source, "lxml")

                tags = soup.find_all(class_="auctions-item-image-link")
                for tag in tags:
                    car_page_links.add(tag.get("href"))
        except:
            print("Couldn't find any more links")
            print("Saving links to csv...")
            links_df = pd.DataFrame(list(car_page_links))
            links_df.to_csv('auction_result_links.csv', index=False)
            print("Saved")
            break


def scrape_oldtimer_auction_results(links, filename='oldtimers.csv'):
    """
    Creates a Dataframe of Oldtimer Auction results

    Args:
        link: (str) the link for an Oldtimer on BringATrailer.com
        filename: (str) the filename to which to save the webscraping results
    Returns:
        dict: containing oldtimer stats including lot number, engine, etc.
    """
    column_names = ["Lot Number", "Chassis Number", "Running Condition", "Seller", "Location",
                    "Model", "Year", "Kilometers", "Engine", "Gearbox", "Date", "Price"]

    oldtimers = pd.DataFrame(columns=column_names)

    iteration = 0
    for link in links:
        oldtimer = Oldtimer(link)
        oldtimers = oldtimers.append(oldtimer.return_dict(), ignore_index=True)
        if iteration % 100 == 0:
            oldtimers.to_csv(filename)
        iteration += 1
    return oldtimers


def clean_text(text):
    """
    Applies some pre-processing on the given text.

    Steps :
    - Removing HTML tags
    - Removing punctuation
    - Lowering text
    """

    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)

    # convert text to lowercase
    text = text.strip().lower()

    # remove special characters
    text = text.replace('\n', '')
    text = text.replace('\xa0', ' ')

    # replace punctuation characters with spaces
    filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text


class Oldtimer:
    """Extracts Oldtimer attributes from BringATrailer.com page

    Args:
        link: (str) the link for which to extract attributes

    """

    def __init__(self, link):
        self.link = link
        self.page = self.set_page()
        self.lot_number = self.find_lot_number()
        self.chassis_number = self.find_chassis_number()
        self.running_condition = self.find_condition()
        self.seller_name = self.find_seller_name()
        self.seller_location = self.find_seller_location()
        self.model = self.find_model()
        self.year = self.find_year()
        self.kilometers = self.find_distance_driven()
        self.engine = self.find_engine()
        self.gearbox = self.find_gearbox()
        self.date = self.find_date_sold()
        self.price = self.find_price()

    def set_page(self):
        """
        Creates a BeautifulSoup page.

        Args:
            None
        Returns:
            None
        """
        try:
            response = requests.get(self.link)
            t = response.text
            return BeautifulSoup(t, "lxml")
        except:
            return None

    def find_lot_number(self):
        """
        Find the object lot Number.

        Args:
            None.
        Returns:
            None
        """
        try:
            lot_number = self.page.find_all(class_="listing-essentials-item")[0].text
            if "Lot #" in lot_number:
                return lot_number.split("#")[1]
            else:
                return np.nan
        except:
            return np.nan

    def find_chassis_number(self):
        """
        Finds the Object Chassis Number.

        Args:
            None
        Returns:
            None.
        """
        try:
            chassis_number = self.page.find_all(class_='listing-essentials-item', text=re.compile("Chassis: "))
            if (len(chassis_number) == 1):
                return chassis_number[0].text.split("Chassis: ")[1]
            elif len(chassis_number > 1):
                return "Multiple Elements Found"
            elif len(car_element < 1):
                return "No elements found"
        except:
            return np.nan

    def find_condition(self):

        try:
            condition_text = self.page.find_all(class_="post-title listing-post-title")[0].text
            if "project" in condition_text.lower():
                return False
            return True
        except:
            return np.nan

    def find_seller_name(self):
        """
        Returns the Car Sellers Name.

        Args:
            None
        Returns:
            None.
        """
        try:
            seller_name = self.page.find_all(class_="listing-essentials-item")[1]
            return seller_name.text.split("Seller: ")[1]
        except:
            return np.nan

    def find_seller_location(self):
        """Identifies seller location on webpage

        Args:
            None
        Returns:
            str: Seller location
        """
        try:
            seller_location = self.page.find_all(class_="listing-essentials-item")[2]
            if len(seller_location) == 1:
                return seller_location.text.split("Location: ")[1]
            elif len(seller_name) > 1:
                return "Multiple Elements Found"
            elif len(seller_name) < 1:
                return "No elements found"
        except:
            return np.nan

    def find_model(self):
        """
        Returns a string of the Model Name and Type given a webpage.

        Args
            None.
        Returns
            A string containing the Model Name and Type

        """
        try:
            model_page = self.page.find(text="Model Page: ")
            if model_page:
                return model_page.findNextSibling().text
            model_page = self.page.find(text="Model Pages: ")
            if model_page:
                return model_page.findNextSibling().text
        except:
            return np.nan

    def find_year(self):
        """
        Returns the year the car was built.

        Args:
            None.
        Returns:
            An Integer of the Year the car was built


        """
        try:
            description = clean_text(self.page.find_all(class_='post-excerpt')[0].text)
            description_words = description.split()

            for word in description_words[:4]:
                year_built = re.findall(r'[0-9]{4,}', word)
                if year_built:
                    return int(year_built[0])
            return None

        except:
            return None

    def find_distance_driven(self):
        """
        Returns a integer containing the kilometers driven.

        Args:
            None
        Returns:
            None

        """
        try:
            distance_element = self.page.find_all(class_='listing-essentials-item',
                                                  text=re.compile('(?i)miles|kilometers'))[0].text

            if any(x in distance_element.lower() for x in ["miles", "kilometers"]):
                if len(re.findall(r"[0-9]+k", distance_element)) >= 1:
                    distance = re.findall(r"[0-9]+k", distance_element)[0]
                    for x in distance_element.split(distance)[1].split():
                        if 'kilometers' in x.lower():
                            return int(distance[:-1]) * 1000
                        elif 'miles' in x.lower():
                            return int(1.60934 * int(distance[:-1])) * 1000
                elif len(re.findall(r"[0-9,]{3,}", distance_element)) >= 1:
                    distance = re.findall(r"[0-9,]{3,}", distance_element)[0]
                    mdistance = distance.replace(',', '')
                    for x in distance_element.split(distance)[1].split():
                        if 'kilometers' in x.lower():
                            return int(mdistance)
                        elif 'miles' in x.lower():
                            return int(1.60934 * int(mdistance))
            else:
                return None
        except:
            return None

    def find_engine(self):
        """
        Returns a string specifying the engine.

        Args:
            page: An lxml parsed BeautifulSoup Webpage
        Returns:
            An integer of the amount of kilometers driven.
        """
        try:
            engine_element = self.page.find_all(class_='listing-essentials-item',
                                                text=re.compile(r'[0-9,]{3,}cc|[0-9\.,]+-L|[0-9]\.[0-9]L| V6| V8'))

            if len(engine_element) == 1:
                return engine_element[0].text
            elif len(engine_element) == 0:
                return "No Engine Data Found"
            else:
                return "Multiple Entries Found"
        except:
            return None

    def find_gearbox(self):
        """
        Returns a string specifying the engine.

        Args:
            None.
        Returns:
            An integer of the amount of kilometers driven.
        """
        try:
            gearbox_element = self.page.find_all(class_='listing-essentials-item',
                                                 text=re.compile(
                                                     r'(?i)^.*Gearbox|Transaxle|Transmission|[0-9]-Speed.*$'))

            return gearbox_element[0].text
        except:
            print("Error!")
            return None

    def find_date_sold(self):
        """
        Returns the date on which the car was sold

        Args:
            page: An lxml parsed BeautifulSoup Webpage
        Returns:
            A datetime object containing the date the car was sold.

        """
        try:
            date_text = self.page.find_all(class_="listing-available-info")[0].text
            date = re.findall(r"[0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4}", date_text)[0]
            return datetime.datetime.strptime(date, '%m/%d/%y')

        except:
            return None

    def find_price(self):
        """
        Returns the price at which the car was sold.

        Args:
            None.
        Returns:
            An Integer of the price of the car in dollars.

        """
        try:
            price_text = self.page.find_all(class_="listing-available-info")[0].text
            price = re.findall(r"\$[0-9,]{3,}", price_text)[0]
            price = price.replace(",", "")
            price = price.replace("$", "")
            return int(price)
        except:
            return None

    def return_dict(self):
        oldtimer_dict = {"Lot Number": self.lot_number, "Chassis Number": self.chassis_number,
                         "Running Condition": self.running_condition,
                         "Seller": self.seller_name, "Location": self.seller_location,
                         "Model": self.model, "Year": self.year, "Kilometers": self.kilometers,
                         "Engine": self.engine, "Gearbox": self.gearbox, "Date": self.date,
                         "Price": self.price}
        return oldtimer_dict
