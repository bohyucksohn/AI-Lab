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

# =============================================================================
# STAGE 3 - ANALYZE
# Prompt: "Run OLS baseline, estimate propensity scores, perform nearest-neighbor 
# matching, compute SMD"
# =============================================================================

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 3a. Baseline OLS
X_ols = sm.add_constant(df_clean[["AI_ADOPTED"]])
ols_model = sm.OLS(df_clean["REV_GROWTH"], X_ols).fit()
print("\n" + "="*70)
print("BASELINE OLS")
print("="*70)
print(ols_model.summary())
print(f"Naive AI coefficient: {ols_model.params['AI_ADOPTED']:.4f}")

# 3b. Propensity Score Estimation
covariates = ["LOG_REV","TEAM_SIZE","FIRM_AGE","RD_SPEND","DIGITAL_SALES","COMPLIANCE_TIER_NUM","FRAUD_EXPOSURE_NUM"]
psm_df = df_clean.dropna(subset=covariates).copy()
X_ps = psm_df[covariates].values
y_ps = psm_df["AI_ADOPTED"].values.astype(int)
logit = LogisticRegression(max_iter=1000, random_state=42)
logit.fit(X_ps, y_ps)
psm_df["PSCORE"] = logit.predict_proba(X_ps)[:, 1]

# Logit summary table
X_logit_sm = sm.add_constant(psm_df[covariates])
logit_sm = sm.Logit(psm_df["AI_ADOPTED"], X_logit_sm).fit(disp=0)
print("\n" + "="*70)
print("PROPENSITY SCORE MODEL")
print("="*70)
print(logit_sm.summary())

# 3c. Common Support Plot
treated = psm_df[psm_df["AI_ADOPTED"]==1]["PSCORE"]
control = psm_df[psm_df["AI_ADOPTED"]==0]["PSCORE"]
fig, ax = plt.subplots(figsize=(8,5))
ax.hist(treated, bins=20, alpha=0.6, label="AI Adopted (Treated)", color="#2196F3", density=True)
ax.hist(control, bins=20, alpha=0.6, label="No AI (Control)", color="#FF9800", density=True)
ax.set_xlabel("Propensity Score"); ax.set_ylabel("Density")
ax.set_title("Common Support: Propensity Score Distribution")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("fig_common_support.png", dpi=200); plt.close()

# 3d. Nearest-Neighbor Matching
treated_idx = psm_df[psm_df["AI_ADOPTED"]==1].index
control_idx = psm_df[psm_df["AI_ADOPTED"]==0].index
nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
nn.fit(psm_df.loc[control_idx, ["PSCORE"]].values)
distances, indices = nn.kneighbors(psm_df.loc[treated_idx, ["PSCORE"]].values)
matched_control_idx = control_idx[indices.flatten()]
matched_df = pd.concat([psm_df.loc[treated_idx].copy(), psm_df.loc[matched_control_idx].copy()], ignore_index=True)
print(f"\nMatched sample: {len(matched_df)} rows ({len(treated_idx)} treated + {len(treated_idx)} controls)")

# 3e. SMD Before and After
def calc_smd(df_t, df_c, col):
    std_pool = np.sqrt((df_t[col].var() + df_c[col].var()) / 2)
    return (df_t[col].mean() - df_c[col].mean()) / std_pool if std_pool != 0 else 0.0

smd_results = []
for cov in covariates:
    smd_b = calc_smd(psm_df[psm_df["AI_ADOPTED"]==1], psm_df[psm_df["AI_ADOPTED"]==0], cov)
    smd_a = calc_smd(matched_df[matched_df["AI_ADOPTED"]==1], matched_df[matched_df["AI_ADOPTED"]==0], cov)
    smd_results.append({"Covariate": cov, "SMD_Before": round(smd_b,4), "SMD_After": round(smd_a,4)})
smd_df = pd.DataFrame(smd_results)
print("\n" + "="*70)
print("SMD TABLE")
print("="*70)
print(smd_df.to_string(index=False))

# 3f. Love Plot
fig, ax = plt.subplots(figsize=(8,6))
y_pos = range(len(smd_df))
ax.scatter(smd_df["SMD_Before"].abs(), y_pos, marker="o", s=80, color="#FF5722", label="Before Matching", zorder=3)
ax.scatter(smd_df["SMD_After"].abs(), y_pos, marker="D", s=80, color="#4CAF50", label="After Matching", zorder=3)
ax.axvline(x=0.1, color="gray", linestyle="--", alpha=0.7, label="SMD = 0.1 threshold")
ax.set_yticks(list(y_pos)); ax.set_yticklabels(smd_df["Covariate"])
ax.set_xlabel("|Standardized Mean Difference|")
ax.set_title("Love Plot: Covariate Balance Before & After Matching")
ax.legend(); ax.grid(axis="x", alpha=0.3)
plt.tight_layout(); plt.savefig("fig_love_plot.png", dpi=200); plt.close()

# 3g. PSM-Adjusted OLS
X_psm = sm.add_constant(matched_df[["AI_ADOPTED"]])
psm_model = sm.OLS(matched_df["REV_GROWTH"], X_psm).fit()
print("\n" + "="*70)
print("PSM-ADJUSTED OLS")
print("="*70)
print(psm_model.summary())
print(f"Naive coeff: {ols_model.params['AI_ADOPTED']:.4f} -> PSM coeff: {psm_model.params['AI_ADOPTED']:.4f}")
