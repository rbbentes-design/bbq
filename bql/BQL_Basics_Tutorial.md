# BQL Basics Tutorial

Published on 09 September 2025 • Updated on 12 January 2026

## Table of Contents
- Introduction
- Foundation of BQL
- The skills you need
- Using BQL in BQuant
- Constructing your target universe
- Retrieving your target data
- Analyzing data with functions
- Working with the output
- Handling errors
- Putting it all together
- Going further

## Introduction

This tutorial is divided into sections that teach the fundamentals of writing and executing BQL queries in BQuant. Each section includes code samples and explanations, as well as code challenges to help you practice what you learn.

**Who this is for:** Quantitative Researchers and Analysts who want to start using BQL in their data analysis.

**What you'll learn:** How to write Python and BQL code to execute basic BQL queries and work with the output. You'll also learn where to find documentation on specific BQL objects and discover next steps in applying BQL to your area of interest.

**Before starting, you should:**
- Have a working knowledge of Python, but you don't need to be an expert.
- Be comfortable navigating and running code in a Jupyter notebook.
- Be familiar with the terms and concepts in the BQL Overview.
- Have a practical understanding of statistics and the financial markets.

## Foundation of BQL

BQL is a powerful tool for requesting data from Bloomberg's vast array of financial data, but its flexibility can make it hard to know where to start. This tutorial will help you write your first BQL queries.

Each BQL query has two required parts: The target universe of securities or other entities you want to analyze, and the target data you want to retrieve about those entities. The result of your query is a matrix (in an output cell, a DataFrame) that is the intersection of your target universe and target data.

To help you build these queries, BQL offers three types of objects: universe functions, data items, and functions. All three of these objects have required and optional parameters that help you further define the exact data you want.

## The Skills You Need

The following sections explain the fundamental skills you'll need to write BQL queries, manipulate their output, and deal with common errors you might encounter.

### Using BQL in BQuant

This topic covers the basics of how the PyBQL library works in BQuant. You'll learn how to set up your environment, the anatomy of a basic query, and what drives the output you get back.

#### About BQL Objects

As mentioned in the previous section, there are three main types of BQL objects you can use to build a query. In BQuant, the PyBQL library provides accessor properties that let you access these BQL objects, assign them to variables, and use them in your requests to the BQL Service:

| BQL Object | PyBQL Property | Action | Example |
|---|---|---|---|
| Universe function | univ | Set criteria to create a target universe of entities. | univ.members() |
| Data item | data | Retrieve a specific data point (e.g., a price) for an entity. | data.px_last() |
| Function | func | Perform a calculation or transformation on a data point. | func.avg() |

#### Setting up your environment

Before you can start retrieving BQL data, you have to set up your environment by following these steps:

1. Import pandas if needed for DataFrame handling.
2. Import Bloomberg's PyBQL library, named bql.
3. Connect to the BQL Service by creating an instance of the bql.Service() class.

```python
# Set up your environment
import pandas as pd
import bql
bq = bql.Service()
```

#### Hello, World

After setting up your environment, you're ready to retrieve BQL data using the following steps:

1. Define your target universe and target data.
2. Create a bql.Request instance with your target universe and target data as arguments.
3. Use the bq.execute() function to send the request to the BQL Service.
4. Print the response as a DataFrame to see the output.

```python
# Set up your environment:
# import pandas as pd
# import bql
# bq = bql.Service()

# Define the target universe of your BQL query: one or more entities
universe = 'AAPL US Equity'

# Define the target data (e.g., last price) you want for the universe
data_item = bq.data.px_last()

# Pass the two parts of your query as arguments for a Request object
# And execute the request
request = bql.Request(universe, data_item)
response = bq.execute(request)

# Parse the response as a DataFrame and print the output
data = response[0].df()
data
```

Output:
```
DATE	CURRENCY	PX_LAST()
ID			
AAPL US Equity	2025-09-09	USD	237.88
```

#### Metadata and output columns

The first thing you might notice about the output is that even though you only asked for the price, you got the value plus two extra columns of data: the date of observation and the currency. This is because most BQL data items include metadata that provides added context to the value.

### Constructing your target universe

You've learned that all BQL queries need a target universe, which consists of one or more entities for which you want to retrieve data. In this section, you'll learn three primary ways to construct a target universe.

#### Using identifiers

Your target universe is the set of securities, indices, countries, or other entities for which you want to retrieve data. The most straightforward way to declare your universe is to list one or more entities by their Bloomberg identifiers (ID).

BQL supports a range of ID types, including tickers, Bloomberg IDs, ISINs, CUSIPs, and more. Regardless of type, IDs are always passed as a string.

**Single ID:**
```python
# Retrieve the company name for the AAPL ticker
universe = 'AAPL US Equity'
data_item = bq.data.name()

request = bql.Request(universe, data_item)
response = bq.execute(request)

data = response[0].df()
data
```

**Multiple IDs:**
```python
# Retrieve the name of three different FI securities
universe = [
    'BY195456 Corp', # Bloomberg ID number
    'US91282CGM73',  # ISIN
    '649519DV4'      # CUSIP
]
data_item = bq.data.name()

request = bql.Request(universe, data_item)
response = bq.execute(request)

data = response[0].df()
data
```

#### Using universe functions

BQL has around 40 universe functions, identifiable by the bq.univ property. Each universe function accepts unique parameters that help you zero in on the entities you want to analyze.

**members() function:**
```python
# Retrieve the names of the current members of the Euro Stoxx 50 Index
universe = bq.univ.members('SX5E Index')
data_item = bq.data.name()

request = bql.Request(universe, data_item)
response = bq.execute(request)

data = response[0].df()
data.head()
```

**Parameters: As-of date:**
```python
# Retrieve the names of the members of the Euro Stoxx 50 Index,
# as of an absolute date
universe = bq.univ.members('SX5E Index', dates='2020-12-31')
data_item = bq.data.name()

request = bql.Request(universe, data_item)
response = bq.execute(request)

data = response[0].df()
data.head()
```

#### Universe filtering

**Conditions: Descriptive data:**
```python
# Retrieve Dow members in the BICS Financials sector
universe = bq.univ.members('INDU Index')
data_item = bq.data.classification_name() 
filtersect = bq.univ.filter(universe, data_item == 'Financials')

request = bql.Request(filtersect, data_item)
response = bq.execute(request)

data = response[0].df()
data
```

**Conditions: Market data:**
```python
# Retrieve Dow members with a last-reported volume above 1,000,000
universe = bq.univ.members('INDU Index')
data_item = bq.data.px_volume() 
filtervol = bq.univ.filter(universe, data_item > 1000000)

request = bql.Request(filtervol, data_item)
response = bq.execute(request)

data = response[0].df()
data
```

### Retrieving your target data

In the previous section, we touched upon a few data items, like name(), classification_name(), and px_last(). In this section, we'll dive deeper into retrieving different kinds of data using BQL data items and their parameters.

#### What is a BQL data item?

In short, a BQL data item is a piece of data that Bloomberg collects or calculates about an entity. It reports a value (e.g., highest price) but also comes with pieces of metadata that provide context, such as a currency or reporting date.

Some broad categories of BQL data items include:
- Descriptive and reference data
- Company financials and analyst estimates
- ESG (Environmental, social, and corporate governance) metrics
- Market data and analytics
- Macro data and economic statistics

#### How to find data items

There are two ways to look up BQL data items:

1. **BQuant Help Center:** If you want complete documentation on any data item, from its signatures and parameters to useful code snippets, check the BQL Data Items page on the BQuant Help Center.
2. **FLDS <GO>:** On the Bloomberg Terminal®, use FLDS <GO>, then select Source > BQL to search for data items.

#### Retrieving single points of data

**Accepting default parameters:**
```python
# Retrieve the last-reported price of Apple's stock
universe = 'AAPL US Equity'
data_item = bq.data.px_last()

request = bql.Request(universe, data_item)
response = bq.execute(request)

data = response[0].df()
data
```

**Dates parameter:**
```python
# Retrieve the last-reported price of Apple's stock on March 6, 2023
universe = 'AAPL US Equity'
data_item = bq.data.px_last(dates='2023-03-06')

request = bql.Request(universe, data_item)
response = bq.execute(request)

data = response[0].df()
data
```

**Combining parameters:**
```python
# Retrieve the last-reported price of Apple's stock on March 6, 2023
# Convert the price to BRL
universe = 'AAPL US Equity'
data_item = bq.data.px_last(dates='2023-03-06', currency='BRL')

request = bql.Request(universe, data_item)
response = bq.execute(request)

data = response[0].df()
data
```

#### Retrieving a time series of data

```python
# Retrieve the last price of Adidas stock for the past 10 days
universe = 'ADS GR Equity'
data_item = bq.data.px_last(dates=bq.func.range('-10d, 0d'))

request = bql.Request(universe, data_item)
response = bq.execute(request)

data = response[0].df()
data
```

### Analyzing data with functions

BQL functions allow you to manipulate and transform data before it's returned to you.

#### Common function types

**Data handling:**
```python
# Retrieve a time series of Vodafone's stock price,
# removing non-trading days
universe = 'VOD LN Equity'
daterange = bq.func.range('2023-02-27', '2023-03-06')
data_item = bq.data.px_last(dates=daterange)
cleaned_data = bq.func.dropna(data_item)

request = bql.Request(universe, cleaned_data)
response = bq.execute(request)

data = response[0].df()
data
```

**Statistical:**
```python
# Retrieve the highest opening price for Vodafone's stock
# during the last 10 days
universe = 'VOD LN Equity'
data_item = bq.data.px_open(dates=bq.func.range('-9d', '0d')).max()

request = bql.Request(universe, data_item)
response = bq.execute(request)

data = response[0].df()
data
```

**Time series manipulation:**
```python
# Retrieve the net change in Vodafone's stock price over the last two weeks
universe = 'VOD LN Equity'
data_item = bq.data.px_last(dates=bq.func.range('-2w', '0d')).net_chg()

request = bql.Request(universe, data_item)
response = bq.execute(request)

data = response[0].df()
data
```

**Grouping:**
```python
# Group the members of the Euro Stoxx 50 Index by country of domicile
universe = bq.univ.members(['SX5E Index'])

data_item = (
    bq.func.group(bq.data.name(), by=bq.data.country_full_name())
    .count()
)

request = bql.Request(universe, data_item)
response = bq.execute(request)

data = response[0].df()
data
```

### Working with the output

#### Multiple data items and value-only output

```python
# Retrieve the price, volume, and EPS for Apple in one DataFrame
universe = 'AAPL US Equity'

# Create a dictionary of the data items
# and ask for the value only (no metadata)
data_items = {
    'Last Price': bq.data.px_last()['value'],
    'Volume': bq.data.px_volume()['value'],
    'EPS': bq.data.is_eps()['value']
}

request = bql.Request(universe, data_items)
response = bq.execute(request)

# Using pandas.concat along with list comprehension,
# concatenate the three values along the columns axis
data = pd.concat([item.df() for item in response],axis=1)
data
```

### Handling errors

There are three main types of errors: query level, item level, and row level.

**Query level error example:**
```python
# INCORRECT - Missing data item
request = bql.Request(universe)
response = bq.execute(request)
```

**Corrected:**
```python
request = bql.Request(universe, data_item)
response = bq.execute(request)
```

**Item level error example:**
```python
# INCORRECT - Data type mismatch
data_item = bq.func.avg(bq.data.exch_code(), bq.data.px_low())
```

**Corrected:**
```python
# Both return a Double
data_item = bq.func.avg(bq.data.px_high(), bq.data.px_low())
```

## Putting it All Together

### Code Challenge 1: Relative Valuation
Compare Apple to similar companies using peers() and is_eps().

### Code Challenge 2: Comp Sheet Creation
Build a comp sheet for FAANG stocks with name, price, market cap, enterprise value, and revenue.

### Code Challenge 3: ESG Analysis
Examine "Green Bonds" in the S&P 500, grouped by BICS sector.

### Code Challenge 4: Technical Analysis
Isolate the top 10 dates over the last 5 years with the largest one-day price moves for Vodafone.

## Going Further

### See also
- **PyBQL API Reference:** Look up documentation on PyBQL and BQL objects on the BQuant Help Center.
- **BQL for Equities Tutorial:** A tutorial teaching you how to use BQL to retrieve data for your research in the equity market.
- **Intro to Fixed Income with BQL:** A notebook showing off a variety of ways you can use BQL to perform fixed income analyses.
- **Visualizations Using Plotly:** Examples showing how to create interactive visualizations using Plotly.

---

*This tutorial provides a comprehensive foundation for working with BQL in BQuant. Continue exploring the BQuant Help Center for more advanced topics and specific use cases.*
