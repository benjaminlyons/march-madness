import bs4
import urllib.request
import re

import pandas as pd
import time
import os
import sys

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
                headers.append("Year")

    return headers

def parse_row(row, year):
    row_data = []
    for index, td in enumerate(row.find_all('td')):
        # if its the school name
        if index == 0: 
            name = td.get_text()
            name = name.removesuffix("NCAA").strip()
            row_data.append(name)
            link = td.find('a').get('href').strip()
            matches = re.search(r'/schools/(.*)/.*.html',link)
            link = matches.group(1)
            row_data.append(link)
            row_data.append(str(year))
        elif index in STAT_INDEX:
            row_data.append(td.get_text())
    return row_data

def scrape_season(year, f, is_first=False):
    url = f"https://www.sports-reference.com/cbb/seasons/{year}-advanced-school-stats.html"

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
        
        data.append(parse_row(row, year))

    if is_first:
        f.write(','.join(headers) + '\n')

    for row in data:
        if row:
            f.write(','.join(row) + '\n')

def scrape_team(teamname, linkname, year, results, schools_scraped):
    url = f"https://www.sports-reference.com/cbb/schools/{linkname}/{year}-gamelogs.html"

    source = urllib.request.urlopen(url)
    soup = bs4.BeautifulSoup(source, 'lxml')

    game_log = soup.find(id='sgl-basic')

    if not game_log:
        return

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
        results.append([teamname, str(year), opp, res, score, opp_score])


def main():
    # with open("all_team_stats.csv", "w") as f:
    #     for year in range(2017, 2023):
    #         if year == 2020:
    #             continue
    #         elif year == 2017:
    #             scrape_season(year, f, True)
    #         else:
    #             scrape_season(year, f)
    #
    df = pd.read_csv("all_team_stats.csv")
    results_headers = ["school1", "year", "school2", "result", "score1", "score2"]
    schools_scraped = set()
    with open('all_game_logs.csv', 'w') as f:
        f.write(",".join(results_headers) + "\n")
        for school, linkname, year in zip(df["School"], df["Link Name"], df["Year"]):
            print(f"Downloading {school} ({year})...", end='')
            sys.stdout.flush()
            results = []
            scrape_team(school, linkname, year, results, schools_scraped)
            schools_scraped.add(school + " " + str(year))
            time.sleep(1)
            for game in results:
                f.write(",".join(game) + "\n")
            f.flush()
            print("Done!")

if __name__ == "__main__":
    main()
