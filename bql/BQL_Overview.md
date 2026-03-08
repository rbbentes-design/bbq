# BQL Overview

This overview introduces key concepts of the Bloomberg Query Language (BQL) and explains the process of requesting and receiving data from the breadth of Bloomberg's financial datasets.

## Table of Contents
- Introduction
- How BQL works
- See also

## Introduction

Building a custom analysis of financial data can be cumbersome. You often need to fetch large, costly datasets, then spend time aligning and manipulating the data before starting your work. BQL is a solution created by and for financial professionals to overcome these barriers.

More than a simple query language, BQL is an API that connects you to the breadth and depth of Bloomberg data, with three main benefits:

### Data modeling
BQL organizes data from many sources into structured datasets, storing uniform metadata for every data point. BQL natively handles many challenges of financial data manipulation — for example, by normalizing fiscal periods and aligning estimates with actuals across datasets.

### Finance-oriented language
BQL is built for financial data and supports multiple asset classes. For example, special operations let you retrieve the members of an index or all active bonds issued by a company, then analyze them individually or as a group.

### Server-side computation
All calculations and data processing are done on Bloomberg servers. For example, this allows you to screen a universe of securities based on multiple properties, all before retrieving any data.

## How BQL works

### Process overview
To access BQL from a requestor application, such as BQuant or Excel, you establish a connection to the BQL Service. In BQuant, this includes importing the BQL library. In Excel, you're automatically connected via the Bloomberg Add-In.

To retrieve BQL data, you write a query in your requestor application, where you define your target universe and target data.

Queries are sent as requests to the BQL Service and on to Bloomberg Servers, where the BQL Calculation Engine fetches data from available BQL Data Sets and transforms it based on your query. It then sends a response back to your requestor application, where the query results appear as an output.

### Queries

Each BQL query contains two required parts: your target universe and your target data.

#### Target universe
Your target universe is the set of securities, indices, countries, or other entities for which you want to retrieve data. One way to declare this universe is by simply listing one or more entities by their Bloomberg identifiers (e.g., 'IBM US Equity', 'SX5E Index').

If you want to work with a larger or more dynamic universe, one way to efficiently isolate it is by using universe functions. For example, you can use the bonds() universe function with the 'AAPL US Equity' identifier to retrieve all bonds issued by Apple.

Universe functions also have required and optional parameters that let you apply further screening (e.g., to retrieve only matured bonds rated B or higher). With these and other filtering operations, you can narrow the scope of your analysis before fetching any data.

#### Target data
To declare your target data, you specify one or more BQL data items out of the hundreds of data sets that Bloomberg stores. A BQL data item represents a value (e.g., highest price) alongside associated columns of metadata, such as the currency and reporting date. The combination of this data item with an entity identifier (ID), such as a security ticker from your target universe, determines the results you get.

Data items return certain data by default, but also offer optional parameters for flexibility. For example, BQL's total_return() data item automatically provides a trailing one-year return. If you want the 30-day return, you simply use total_return() with a 30-day parameter and BQL calculates it on demand.

Similar to how universe functions help you define a universe, you can use regular functions to perform analysis on data items. Available functions range from averages and aggregation to Z-scores and more. Functions also have required and optional parameters that let you create a more customized calculation, such as a rolling one-month average, all computed on the server side.

### Calculation

For the most basic queries, such as one data item for one security, the BQL Calculation Engine simply fetches the data from available BQL Data Sets, using the data item's defaults to produce meaningful results.

However, if your query involves any screening or other customizations, the BQL Calculation Engine works through them to compute results based on your criteria. For example, it might:

- Apply universe functions in order to isolate a universe.
- Apply parameters to data items in order to customize the data.
- Apply functions to data items in order to perform calculations or other analysis on the data.

### Output

The BQL Service sends the output of your query as a response to the requestor application. The exact format of the results can vary based on optional display preferences you include.

In Excel, multiple data items are automatically shown in a single table if they share a metadata column. In BQuant, you can concatenate individual data items into a single DataFrame.

## See also
- PyBQL API Reference - Browse the Python library that allows you to interact with BQL in BQuant, including all BQL data items and functions.
- BQL Basics Tutorial - Learn how to write and execute BQL queries using Python in BQuant.
