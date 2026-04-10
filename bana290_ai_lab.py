import requests, pandas as pd
from bs4 import BeautifulSoup

# =============================================================================
# STAGE 1 - SCRAPE
# Prompt: "Scrape the HTML table of firm profiles from the BANA290 assignment 
# page using BeautifulSoup"
# =============================================================================

url = "https://bana290-assignment1.netlify.app/"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
table = soup.find("table")
rows = table.find_all("tr")

# Extract headers from first row
headers = [td.get_text(strip=True) for td in rows[0].find_all("td")]

# Extract data rows - use <strong> tag for firm names to avoid nested metadata
data = []
for row in rows[1:]:
    cells = row.find_all("td")
    if len(cells) < len(headers):
        continue
    row_data = []
    for i, cell in enumerate(cells):
        if i == 0:
            strong = cell.find("strong")
            row_data.append(strong.get_text(strip=True) if strong else cell.get_text(strip=True))
        else:
            row_data.append(cell.get_text(strip=True))
    data.append(row_data)

df_raw = pd.DataFrame(data, columns=headers)
print(f"Scraped {len(df_raw)} firm profiles with {len(headers)} columns")
print(df_raw.head(3))
