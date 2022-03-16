import bs4
import urllib.request
import re

import pandas as pd
import time

ADVANCED_STATS = set(["School", "W-L%", "SRS", "SOS", "Pace", "ORtg", "FTr", "3PAr", "TS%", "TRB%", "AST%", "STL%", "BLK%", "eFG%", "TOV%", "ORB%"]) # these are the stats we care about
STAT_INDEX = set()

def parse_headers(row):
    headers = []

    for index, th in enumerate(row.find_all('th')):
        if th.get_text().strip() in ADVANCED_STATS:
            headers.append(th.get_text())
            STAT_INDEX.add(index-1)
            if index == 1:
                headers.append("Link Name")

    return headers

def parse_row(row):
    row_data = []
    for index, td in enumerate(row.find_all('td')):
        # if its the school name
        if index == 0: 
            name = td.get_text()
            name = name.removesuffix("NCAA").strip()
            row_data.append(name)
            link = td.find('a').get('href').strip()
            matches = re.search(r'/schools/(.*)/2022.html',link)
            link = matches.group(1)
            row_data.append(link)
        elif index in STAT_INDEX:
            row_data.append(td.get_text())
    return row_data

def scrape_season():
    url = "https://www.sports-reference.com/cbb/seasons/2022-advanced-school-stats.html"

    source = urllib.request.urlopen(url)
    soup = bs4.BeautifulSoup(source, 'lxml')

    stat_table = soup.find(id='adv_school_stats')

    data = []

    for index, row in enumerate(stat_table.find_all("tr")):
        if index == 0:
            continue

        if index == 1:
            headers = parse_headers(row)
            continue
        
        data.append(parse_row(row))

    with open('team_stats.csv', "w") as f:
        f.write(','.join(headers) + '\n')

        for row in data:
            if row:
                f.write(','.join(row) + '\n')

def scrape_team(teamname, linkname, results, schools_scraped):
    url = f"https://www.sports-reference.com/cbb/schools/{linkname}/2022-gamelogs.html"

    source = urllib.request.urlopen(url)
    soup = bs4.BeautifulSoup(source, 'lxml')

    game_log = soup.find(id='sgl-basic')

    games = []
    for index, row in enumerate(game_log.find_all('tr')):
        opp = row.find('td', attrs={'data-stat': 'opp_id'})
        if not opp:
            continue

        opp = opp.get_text()

        # if we already scraped games from this school
        # if opp in schools_scraped:
        #     continue

        res = row.find('td', attrs={'data-stat': 'game_result'}).get_text()

        # if its an L set it to 0, otherwise, set it to 1
        if res == "L":
            res = '0'
        else: 
            res = '1'

        score = row.find('td', attrs={'data-stat': 'pts'}).get_text()
        opp_score = row.find('td', attrs={'data-stat': "opp_pts"}).get_text()
        results.append([teamname, opp, res, score, opp_score])


def main():
    scrape_season()

    df = pd.read_csv("team_stats.csv")
    results_headers = ["school1", "school2", "result", "score1", "score2"]
    results = []
    schools_scraped = set()
    for school, linkname in zip(df["School"], df["Link Name"]):
        print(f"Downloading {school}...", end='')
        scrape_team(school, linkname, results, schools_scraped)
        schools_scraped.add(school)
        time.sleep(1)
        print("Done!")

    with open('game_logs.csv', 'w') as f:
        f.write(",".join(results_headers) + "\n")
        for game in results:
            f.write(",".join(game) + "\n")

if __name__ == "__main__":
    main()
