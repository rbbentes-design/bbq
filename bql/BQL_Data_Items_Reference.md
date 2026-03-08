# BQL Data Items Reference

Complete reference of BQL data items available in BQuant.

**Total Items Available:** 19,597+

## Financial Statement Data

### Income Statement Items
- `is_eps()` - Earnings Per Share (EPS)
- `is_basic_eps()` - Basic EPS
- `is_diluted_eps()` - Diluted EPS
- `is_basic_eps_cont_ops()` - EPS Adjusted – Basic
- `earn_for_common()` - Net Income Available to Common Shareholders
- `is_net_income()` - Net Income
- `is_revenues()` - Total Revenue
- `sales_rev_turn()` - Revenue
- `is_cogs_to_fe_and_pp_and_g()` - Cost of Revenue
- `is_ebitda()` - EBITDA
- `is_ebit()` - EBIT
- `is_inc_tax_exp()` - Income Tax Expense (Benefit)
- `is_net_interest_expense()` - Net Interest Expense
- `pretax_inc()` - Pre-tax Income (PTP)

### Margin & Profitability Metrics
- `ebitda_margin()` - EBITDA Margin (%)
- `ebit_margin()` - Trailing 12M EBIT Margin
- `net_profit_margin()` - Net Profit Margin
- `gross_profit_margin()` - Gross Profit Margin

### Balance Sheet Items
- `bs_sh_out()` - Shares Outstanding
- `bs_st_borrow()` - Short-Term Debt Including Leases Liabilities
- `bs_inventories()` - Inventories
- `net_debt()` - Net Debt

### Cash Flow Items
- `cf_dvd_paid()` - Dividends Paid
- `free_cash_flow_per_sh()` - Free Cash Flow per Share – Basic
- `free_cash_flow_yield()` - Free Cash Flow Yield

## Equity Data

### Pricing Data
- `px_last()` - Last Price
- `px_open()` - Open Price
- `px_high()` - High Price
- `px_low()` - Low Price
- `px_bid()` - Bid Price
- `px_ask()` - Ask Price
- `px_volume()` - Trading Volume
- `px_close()` - Close Price

### Valuation Metrics
- `cur_mkt_cap()` - Market Capitalization
- `curr_entp_val()` - Enterprise Value
- `pe_ratio()` - Price-to-Earnings Ratio
- `pb_ratio()` - Price-to-Book Ratio
- `ps_ratio()` - Price-to-Sales Ratio
- `pcf_ratio()` - Price-to-Cash Flow Ratio

### Returns Data
- `total_return()` - Total Return
- `return_series()` - Return Series
- `dividend_yield()` - Dividend Yield

### Share Data
- `is_avg_num_sh_for_eps()` - Weighted Average Number of Shares – Basic
- `is_sh_for_diluted_eps()` - Diluted Weighted Average Shares
- `eqy_free_float_pct()` - Free Float Percent

## Fixed Income Data

### Pricing & Spreads
- `spread()` - Option-Adjusted Spread (OAS)
- `spread_to_mid_swaps_at_issue()` - Spread to Mid Swaps at Issue
- `yield()` - Yield
- `ytm()` - Yield to Maturity
- `duration()` - Duration

### Credit Data
- `rating()` - Credit Rating
- `rating_source` - Rating Agency (S&P, Moody's, Fitch)

### Bond Data
- `amt_issued()` - Amount Issued
- `coupon()` - Coupon Rate
- `maturity_date()` - Maturity Date

## Descriptive Data

### Company Info
- `name()` - Entity Name
- `country_full_name()` - Country Full Name
- `sector()` - Sector
- `bics_level_1_sector_name()` - BICS Level 1 Sector Name
- `classification_name()` - Classification Name

### Fundamental Properties
- `eqy_fund_crncy()` - Fundamental Currency

### Corporate Actions
- `dividend()` - Dividend
- `earnings()` - Earnings

## ESG Data

### Environmental
- `greenhouse_gas_emissions()` - GHG Emissions
- `renewable_energy_pct()` - Renewable Energy Percentage

### Social
- `pct_women_employees()` - Percentage Women Employees
- `employee_diversity()` - Employee Diversity Metrics

### Governance
- `board_independence_pct()` - Board Independence Percentage

## Interest Rate & Economic Data

### Macro Indicators
- `cpi()` - Consumer Price Index
- `gdp()` - Gross Domestic Product
- `unemployment()` - Unemployment Rate
- `inflation_rate()` - Inflation Rate

### Interest Rates
- `interest_rate()` - Interest Rate
- `yield_curve()` - Yield Curve

## Derivatives Data

### Options Data
- `implied_volatility()` - Implied Volatility
- `open_int()` - Open Interest
- `option_chain()` - Option Chain

## Using Data Items

### Basic Syntax
```python
import bql
bq = bql.Service()

# Simple data item access
data_item = bq.data.px_last()

# Data item with parameters
data_item = bq.data.is_eps(fa_period_type='q')  # Quarterly EPS

# Data item with dates
data_item = bq.data.px_last(dates='2023-01-01')
```

### Common Parameters
- `dates` - Specify historical dates
- `currency` - Convert currency
- `fa_period_type` - Period type (LTM, A, Q, M)
- `periodicity` - Data frequency
- `rating_source` - Rating agency

## Finding More Items

1. **BQuant Help Center:** Visit the BQL Data Items page for searchable documentation
2. **Bloomberg Terminal:** Use `FLDS <GO>` and select Source > BQL
3. **PyBQL Documentation:** Refer to the bql.Service.data section
4. **Contact Bloomberg:** For custom or specialized data items

---

*This is a reference guide. For complete details and parameter options for each data item, refer to the BQuant Help Center or use the help() function in BQuant.*
