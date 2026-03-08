# PyBQL API Reference

PyBQL is a Python library that allows you to interact with the Bloomberg Query Language (BQL) from within the BQuant Platform. BQL offers a comprehensive suite of over 10,000 data items and hundreds of built-in functions, giving you the tools to build complex queries and retrieve financial data for analysis.

## Classes and Functions

The PyBQL library contains the following classes and functions, which allow you to connect to the BQL service, generate and execute requests, and format responses:

### bql.BqlItem

The implementation of a BQL universe, data item, or function.

### bql.combined_df

Return a pandas.DataFrame that combines the results from multiple SingleItemResponse objects.

### bql.let

Create a new Let Item (_LetItem) with the given name and expression.

### bql.Request

Create a BQL Request object.

### bql.Response

The result of executing a BQL Request.

### bql.Service

The primary entry point for BQL interaction.

### bql.SingleItemResponse

Represents BQL's response to a single data item in a Request.

---

## Usage Overview

To use PyBQL in BQuant:

1. Import the BQL library
2. Create a Service instance
3. Build your queries using BqlItem, universe functions, and data items
4. Execute requests with Request objects
5. Handle responses with Response and SingleItemResponse objects
6. Combine results using combined_df for analysis

For detailed examples and tutorials, see BQL Basics Tutorial.
