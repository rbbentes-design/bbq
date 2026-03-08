# Querying Custom Data from CDE

> Updated on 12 January 2026

This page explains how to query and filter custom non-BQL data stored in the Custom Data Editor (CDE \<GO\>) function on the Bloomberg Terminal® using PyBQL.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Set up your environment](#set-up-your-environment)
- [Query a custom data field](#query-a-custom-data-field)
- [View daily history](#view-daily-history)
- [Filter a universe with CDE fields](#filter-a-universe-with-cde-fields)
- [Query a long text entry](#query-a-long-text-entry)

---

## Overview

Custom data you have access to in CDE \<GO\> is also available in BQuant through the PyBQL `bql.Service.data._cde()` method. This means that you can query the custom data that you use in Bloomberg Terminal® functions, such as Portfolio & Risk Analytics (PORT \<GO\>) and My Graphs & Maps (G \<GO\>) in BQuant as well.

> **Note:** You cannot upload custom data from BQuant to CDE \<GO\>. This means that custom data values that are changed in BQuant can only be used in BQuant. You cannot use these new custom data values in Terminal functions.

Regardless of the type of data that is stored in your custom data field, the procedure for querying the data is the same.

**To query data from CDE \<GO\>:**

1. Set up your environment for querying custom data from CDE \<GO\>.
2. Retrieve the field mnemonic names for your custom data fields using CDE \<GO\>.
3. Retrieve custom data using the field mnemonic names and `bq.data._cde`.

> **Note:** Field mnemonic names are the human-readable identifiers for your custom data fields. When you create a custom data field in CDE \<GO\>, you enter the field mnemonic name you want to use for the field.

---

## Prerequisites

Before you can query custom data, you must have access to the custom data fields you want to use in CDE \<GO\>. If you are querying a colleague's custom data, your colleague must share the custom data field to your User ID (UUID).

For instructions about sharing a custom data field, see **Sharing a Field on the Terminal**.

> **Note:** The code examples on this page assume you are familiar with the PyBQL API. For information about PyBQL, see the PyBQL API Reference.

---

## Set up your environment

To set up your environment for querying custom data from CDE \<GO\>:

1. Record the field mnemonic names for your custom data fields from CDE \<GO\>.
2. In a BQuant notebook, set up required packages, variables, and the BQL Service:

```python
import pandas as pd
import bql
import os

bq = bql.Service()
```

---

## Query a custom data field

The code example in this section assumes that you have already added data to a custom data field with a **Number** content type in CDE \<GO\>. In the example, the field mnemonic name for this field is `UD_MODELSTORE`. When you run the query in the code example, the output will reflect the data you created in CDE \<GO\> instead of the example output.

**To query a custom data field:**

1. Ensure that you completed [Set up your environment](#set-up-your-environment) above.
2. Query the custom data field:

```python
universe = 'AAPL US Equity'

with_params = dict(fill='prev', dates='2019-06-11')

# Assign a field mnemonic name from the Set up your environment step to a variable
modelstore_field = f'UD_MODELSTORE'

# Retrieve data from a custom data field in CDE
item = bq.data._cde(modelstore_field)

request = bql.Request(universe, item, with_params)
response = bq.execute(request)
df_modelstore = response[0].df()

df_modelstore.head()
```

**Example Output:**

| | Date | UD_MODELSTORE() |
|---|---|---|
| **ID** | | |
| AAPL US Equity | 2019-06-11 | 0.1908 |

---

## View daily history

The code example in this section assumes that you have already added time series data to a custom data field with a **Number** content type in CDE \<GO\>. In the example, the field mnemonic name for this field is `UD_HALO`. When you run the query in the code example, the output will reflect the data you created in CDE \<GO\> instead of the example output.

**To view the daily history for custom data:**

1. Ensure that you completed [Set up your environment](#set-up-your-environment) above.
2. Query the custom data field for specified date range:

```python
universe = 'SAP GY Equity'

start_date = '2016-01-01'
end_date = '2017-12-31'

# Create a date range for daily history
date_range = bq.func.range(start=start_date, end=end_date, frq='d')
with_params = dict(dates=date_range, fill='prev')

# Assign a field mnemonic name from the Set up your environment step to a variable
halo_field = f'UD_HALO'

# Retrieve data from a custom data field in CDE
item = bq.data._cde(halo_field).dropna()

request = bql.Request(universe, item, with_params)
response = bq.execute(request)
df_sap = bql.combined_df(response)

df_sap.head()
```

**Example Output:**

| | Date | DROPNA(UD_HALO()) |
|---|---|---|
| **ID** | | |
| SAP GY Equity | 2016-01-01 | 73.38 |
| SAP GY Equity | 2016-01-02 | 73.38 |
| SAP GY Equity | 2016-01-03 | 73.38 |
| SAP GY Equity | 2016-01-04 | 70.58 |
| SAP GY Equity | 2016-01-05 | 71.43 |

---

## Filter a universe with CDE fields

The code example in this section assumes that you have already added point-in-time data to a custom data field with a **Date** content type in CDE \<GO\>. In the example, the field mnemonic name for this field is `UD_CUSTOM_UNIV_DATES`. When you run the query in the code example, the output will reflect the data you created in CDE \<GO\> instead of the example output.

**To filter a universe using custom data fields:**

1. Ensure that you completed [Set up your environment](#set-up-your-environment) above.
2. Filter a universe with point-in-time custom data:

```python
universe = bq.univ.members('INDU Index')

date = '2015-01-31'

# Assign a field mnemonic name from the Set up your environment step to a variable
custom_univ_date = f'UD_CUSTOM_UNIV_DATES'

# Retrieve point-in-time custom data from CDE
item = bq.data._cde(custom_univ_date, dates=date)

# Filter your universe by the point-in-time DataFrame
filtered_univ = universe.filter(item != bql.NA)
request = bql.Request(filtered_univ, item)
response = bq.execute(request)
df_univ_dates = response[0].df()

df_univ_dates.head()
```

**Example Output:**

| | Date | UD_CUSTOM_UNIV_DATES(dates=2015-01-31) |
|---|---|---|
| **ID** | | |
| AXP UN Equity | 2015-01-31 | 2015-01-31 |
| VZ UN Equity | 2015-01-31 | 2015-01-31 |
| BA UN Equity | 2015-01-31 | 2015-01-31 |
| CAT UN Equity | 2015-01-31 | 2015-01-31 |
| JPM UN Equity | 2015-01-31 | 2015-01-31 |

---

## Query a long text entry

The code example in this section assumes that you have already added data to a custom data field with a **Long Text** content type in CDE \<GO\>. In the example, the field mnemonic name for this field is `UD_LONGTEXT`. When you run the query in the code example, the output will reflect the data you created in CDE \<GO\> instead of the example output.

**To query a custom data field with long text:**

1. Ensure that you completed [Set up your environment](#set-up-your-environment) above.
2. Query the custom data field:

```python
universe = 'AAPL US Equity'

with_params = dict(fill='prev', dates='2019-06-11')

# Assign a field mnemonic name from the Set up your environment step to a variable
longtext_field = f'UD_LONGTEXT'

# Retrieve the Long Text custom data from CDE
item = bq.data._cde(longtext_field)

request = bql.Request(universe, item, with_params)
response = bq.execute(request)
df_longtext = bql.combined_df(response)

df_longtext.head()
```

**Example Output:**

| | Date | UD_LONGTEXT() |
|---|---|---|
| **ID** | | |
| AAPL US Equity | 2019-06-11 | Apple Inc. designs, manufactures, and markets, … |
