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

# =============================================================================
# STAGE 2 - CLEAN
# Prompt: "Clean the scraped dataframe: parse revenue strings to floats, 
# standardize AI adoption to binary, handle missing values"
# =============================================================================

import numpy as np, re

col_map = {"Firm":"FIRM","Segment":"SEGMENT","HQ Region":"HQ_REGION","Founded":"FOUNDED",
    "Team Size":"TEAM_SIZE","Annual Rev.":"ANNUAL_REV","Rev Growth (YoY)":"REV_GROWTH",
    "R&D Spend":"RD_SPEND","AI Program":"AI_STATUS","Cloud Stack":"CLOUD_STACK",
    "Digital Sales":"DIGITAL_SALES","Compliance Tier":"COMPLIANCE_TIER",
    "Fraud Exposure":"FRAUD_EXPOSURE","Funding Stage":"FUNDING_STAGE","Customer Accts":"CUSTOMER_ACCTS"}
df = df_raw.rename(columns=col_map).copy()

def parse_revenue(val):
    if pd.isna(val) or val.strip() == "": return np.nan
    s = val.lower().strip().replace("usd","").replace("$","").replace(",","").strip()
    multiplier = 1
    if "million" in s: s = s.replace("million","").strip(); multiplier = 1_000_000
    elif "mn" in s: s = s.replace("mn","").strip(); multiplier = 1_000_000
    elif s.endswith("m"): s = s[:-1].strip(); multiplier = 1_000_000
    elif s.endswith("k"): s = s[:-1].strip(); multiplier = 1_000
    try: return float(s) * multiplier
    except ValueError: return np.nan

def parse_pct(val):
    if pd.isna(val) or val.strip() in ("","--","N/A","Unknown"): return np.nan
    s = val.strip().replace("+","").replace("%","").strip()
    try: return float(s)
    except ValueError: return np.nan

def parse_rd_spend(val, annual_rev):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if s in ("","--","n/a","unknown","nan"): return np.nan
    if "% rev" in s or "%rev" in s:
        pct_str = re.sub(r"[^\d.]", "", s.split("%")[0])
        try:
            pct = float(pct_str)
            if pd.notna(annual_rev): return annual_rev * pct / 100.0
        except ValueError: pass
        return np.nan
    s = s.replace("usd","").replace("$","").replace(",","").strip()
    multiplier = 1
    if "million" in s: s = s.replace("million","").strip(); multiplier = 1_000_000
    elif "mn" in s: s = s.replace("mn","").strip(); multiplier = 1_000_000
    elif s.endswith("m"): s = s[:-1].strip(); multiplier = 1_000_000
    elif s.endswith("k"): s = s[:-1].strip(); multiplier = 1_000
    try: return float(s) * multiplier
    except ValueError: return np.nan

def standardize_ai(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if s in ("","--","n/a","unknown"): return np.nan
    if s in ("yes","adopted","ai enabled","production","live"): return 1
    if s in ("no","not yet","legacy only","manual only"): return 0
    if s in ("in review","pilot"): return np.nan
    return np.nan

def parse_team_size(val):
    s = str(val).strip().lower().replace(",","")
    if "k" in s:
        try: return float(s.replace("k","").strip()) * 1000
        except ValueError: return np.nan
    try: return float(s)
    except ValueError: return np.nan

def parse_customers(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower().replace(",","")
    multiplier = 1
    if s.endswith("k"): s = s[:-1].strip(); multiplier = 1_000
    elif s.endswith("m"): s = s[:-1].strip(); multiplier = 1_000_000
    try: return float(s) * multiplier
    except ValueError: return np.nan

df["ANNUAL_REV"] = df["ANNUAL_REV"].apply(parse_revenue)
df["REV_GROWTH"] = df["REV_GROWTH"].apply(parse_pct)
df["RD_SPEND"] = df.apply(lambda r: parse_rd_spend(r["RD_SPEND"], r["ANNUAL_REV"]), axis=1)
df["AI_ADOPTED"] = df["AI_STATUS"].apply(standardize_ai)
df["TEAM_SIZE"] = df["TEAM_SIZE"].apply(parse_team_size)
df["FOUNDED"] = pd.to_numeric(df["FOUNDED"], errors="coerce")
df["DIGITAL_SALES"] = df["DIGITAL_SALES"].apply(parse_pct)
df["CUSTOMER_ACCTS"] = df["CUSTOMER_ACCTS"].apply(parse_customers)
df["COMPLIANCE_TIER_NUM"] = df["COMPLIANCE_TIER"].map({"Tier 1":1,"Tier 2":2,"Tier 3":3,"Tier 4":4})
df["FRAUD_EXPOSURE_NUM"] = df["FRAUD_EXPOSURE"].map({"Low":1,"Moderate":2,"Elevated":3,"High":4})
df["FIRM_AGE"] = 2026 - df["FOUNDED"]
df["LOG_REV"] = np.log1p(df["ANNUAL_REV"])

df_clean = df.dropna(subset=["REV_GROWTH","AI_ADOPTED","ANNUAL_REV","TEAM_SIZE","DIGITAL_SALES","FIRM_AGE","RD_SPEND"]).copy()
print(f"\nRows after cleaning: {len(df_clean)} (dropped {len(df)-len(df_clean)})")
print(df_clean["AI_ADOPTED"].value_counts())
