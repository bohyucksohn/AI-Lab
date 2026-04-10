# AI Lab: Propensity Score Matching Analysis

**BANA 290 — AI Impact Assignment**

## Overview

This project investigates whether AI adoption causally impacts year-over-year revenue growth among 114 North American fintech firms. Using web scraping, data cleaning, and Propensity Score Matching (PSM), the analysis separates the true effect of AI from selection bias driven by firm size, R&D investment, and digital maturity.

## Key Findings

| Metric | Value |
|--------|-------|
| Naive OLS coefficient | 5.58 pp |
| PSM-adjusted coefficient | 3.52 pp |
| Bias reduction | ~37% |

The naive estimate overstated AI's impact by roughly 37%. After matching AI adopters with observationally similar non-adopters, the effect remains positive and statistically significant but is substantially smaller — indicating that larger, higher-spending firms are more likely to both adopt AI and grow faster, independent of AI itself.

## Repository Structure

```
AI-Lab/
├── bana290_ai_lab.py          # Main script (Scrape → Clean → Analyze → Interpret)
├── interpretation.tex         # LaTeX report (12pt, 1in margins, 1.5 spacing)
├── fig_common_support.png     # Propensity score distribution plot
├── fig_love_plot.png          # SMD balance (Love Plot) before/after matching
└── README.md
```

## Commit History

| Commit | Stage | Description |
|--------|-------|-------------|
| 1 | Scrape | Extract 114 firm profiles from HTML table using BeautifulSoup |
| 2 | Clean | Standardize revenue, AI status, R&D spend; drop incomplete rows |
| 3 | Analyze | OLS baseline, propensity scores, nearest-neighbor matching, SMD |
| 4 | Interpret | Add LaTeX report with figures, tables, and PSM interpretation |

## How to Run

```bash
pip install requests beautifulsoup4 pandas numpy statsmodels scikit-learn matplotlib seaborn
python bana290_ai_lab.py
```

The script scrapes live data from [the assignment page](https://bana290-assignment1.netlify.app/), cleans it, runs the full PSM pipeline, and saves two figures to the working directory.

## Tools Used

- **GitHub Copilot** — primary coding assistant within VSCode
- **Python 3** — BeautifulSoup, pandas, statsmodels, scikit-learn, matplotlib
- **LaTeX** — interpretation report formatted per assignment specifications

## Data Source

[North American Fintech & Financial Services Directory](https://bana290-assignment1.netlify.app/) — simulated Q1 2026 filing cycle with 114 firm profiles across 7 segments.
