# Get Data in BQuant

Published on 02 February 2026

## Overview

In BQuant, your main way to access Bloomberg's financial datasets is through the Bloomberg Query Language (BQL). BQL is a powerful query language that lets you retrieve, manipulate, and analyze data across equities, fixed income, funds, commodities, and other datasets. BQuant uses the Python implementation of BQL, the PyBQL library.

This section explains how to access and work with data in BQuant, from understanding what data is available to running your first queries and managing your custom datasets.

## Get Data Resources

### Available Data
Explore the data available to you in BQuant. This includes a searchable list of available data items in BQL that represent fields of BQL datasets, in addition to specialized datasets like Bloomberg's Second Measure consumer transaction data.

### Data Management
Learn how to work with data in BQuant, like accessing your Bloomberg portfolios and querying custom data in the Bloomberg Terminal®.

### BQL Quickstarts
Jump straight into BQL with practical code snippets that query:
- Equities
- Fixed income
- Funds
- FX
- Economics
- Sustainable finance
- Commodities
- Alternative data

### Learn BQL
Learn BQL fundamentals through structured tutorials:
- **BQL Basics:** Understand query syntax, data items, and functions
- **BQL for Equities Tutorial:** Gain greater depth in equities-specific analysis

### BQL Documentation
Access BQL reference documentation, including the PyBQL API reference for detailed function and method documentation.

---

## BQL Data Items

BQL contains thousands of data items drawn from a wide array of Bloomberg datasets. You can access a range of these BQL data items using the PyBQL library. The data items you can access are dependent on if you are using BQuant Desktop or BQuant Enterprise.

For information on using BQL data items, functions, and universe functions, refer to the `bql.Service` section of the PyBQL API Reference.

### Explore Data Items

To find documentation for specific BQL data items, filter the table by name or description.

**Total available items: 19,597+**

#### Example Data Items

| Data Item | Description |
|---|---|
| pretax_inc() | Pre-tax Income (PTP) |
| open_int() | Open Interest |
| is_sh_for_diluted_eps() | Diluted Weighted Average Shares |
| eqy_free_float_pct() | Free Float Percent |
| ebitda_margin() | EBITDA Margin (%) |
| minority_noncontrolling_interest() | Non-Controlling Interest |
| is_avg_num_sh_for_eps() | Weighted Average Number of Shares – Basic |
| bs_st_borrow() | Short-Term Debt Including Leases Liabilities |
| cf_dvd_paid() | Dividends Paid |
| bs_sh_out() | Shares Outstanding |
| net_debt() | Net Debt |
| amt_issued() | Amount Issued |
| earn_for_common() | Net Income Available to Common Shareholders |
| is_inc_tax_exp() | Income Tax Expense (Benefit) |
| px_bid() | Bid Price |
| is_cogs_to_fe_and_pp_and_g() | Cost of Revenue |
| free_cash_flow_per_sh() | Free Cash Flow per Share – Basic |
| ebit_margin() | Trailing 12M EBIT Margin |
| eqy_fund_crncy() | Fundamental Currency |
| spread_to_mid_swaps_at_issue() | Spread to Mid Swaps at Issue |
| country_full_name() | Country Full Name |
| bs_inventories() | Inventories |
| is_net_interest_expense() | Net Interest Expense |
| is_basic_eps_cont_ops() | EPS Adjusted – Basic |

### Accessing Data Items

Data items are accessed through the PyBQL library using the `bq.data` prefix:

```python
import bql
bq = bql.Service()

# Example: Access the last price data item
last_price = bq.data.px_last()

# Example: Access pre-tax income
pretax_income = bq.data.pretax_inc()

# Example: Access shares outstanding
shares_out = bq.data.bs_sh_out()
```

### Data Item Categories

BQL data items cover various financial data categories:

- **Market Data:** Prices, volumes, yields
- **Fundamental Data:** Financial statements, ratios, metrics
- **Descriptive Data:** Names, classifications, identifiers
- **Fixed Income Data:** Spreads, ratings, durations
- **Economic Data:** GDP, inflation, employment figures
- **ESG Data:** Environmental, social, governance metrics
- **Alternative Data:** Consumer transactions, ESG sentiment
- **Derivatives Data:** Options, implied volatility
- **FX Data:** Exchange rates, forward rates

### Finding Data Items

**Method 1: BQuant Help Center**
Search for data items on the BQL Data Items page with a searchable list of all available items.

**Method 2: Bloomberg Terminal**
Use `FLDS <GO>` on the Bloomberg Terminal:
1. Select Source > BQL
2. Enter keywords to find matching data items

**Method 3: PyBQL Library**
Use tab completion and help functions in BQuant:
```python
# View available data items
bq.data.px_last.__doc__

# Use help
help(bq.data.px_last)
```

### Data Availability

The data items you can access depend on:
- Your Bloomberg subscription
- Whether you're using BQuant Desktop or BQuant Enterprise
- Dataset access rights

Consult your Bloomberg representative for information on your data access.

---

## Next Steps

1. **Start with BQL Basics Tutorial** - Learn fundamental query syntax
2. **Explore Available Data** - Search for data items relevant to your analysis
3. **Run BQL Quickstarts** - See practical examples for your asset class
4. **Access PyBQL API Reference** - Get detailed documentation on all functions

