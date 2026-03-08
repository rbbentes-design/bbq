# Accessing Your Portfolios in BQuant

> Published on 05 January 2026 • Updated on 04 March 2026

This article explains the key differences between using BQL functions and the Portfolio library to access portfolio holdings in BQuant.

## Table of Contents

- [Overview](#overview)
- [BQL functions: univ.holdings() and univ.members()](#bql-functions-univholdings-and-univmembers)
- [Portfolio library](#portfolio-library)
- [Feature breakdown](#feature-breakdown)
- [Discovering portfolios](#discovering-portfolios)
- [Date control and history](#date-control-and-history)
- [Weights and normalization](#weights-and-normalization)
- [Zero positions](#zero-positions)
- [Cash representation](#cash-representation)
- [Getting BQL-like results from the Portfolio library](#getting-bql-like-results-from-the-portfolio-library)
- [See also](#see-also)

---

## Overview

BQuant offers multiple ways to access portfolio holdings. BQL functions such as `univ.holdings()` and `univ.members()` provide a quick way to get a snapshot of a portfolio's constituents for analysis and reporting. The Portfolio library is a more comprehensive API that supports programmatic discovery and richer reads of portfolio data, such as retrieving full historical holdings data.

---

## BQL functions: univ.holdings() and univ.members()

The BQL methods `univ.holdings()` and `univ.members()` are designed to retrieve positions for an ETF, mutual fund, market index, or institutional holdings. Additionally, both functions support retrieving the constituent components of a given portfolio.

> **Note:** Both BQL and the Portfolio API return security tickers consistent with the PRTU \<GO\> function on the Bloomberg Terminal®.

These functions can give you the current membership of a portfolio as of today, but they are limited in their ability to fetch historical data:

- **`univ.members()`** — Can only fetch the most recent holdings information.
- **`univ.holdings()`** — More flexible; can be configured to return data for any single historical date, provided the portfolio is a PORT-E (Enterprise) portfolio.

As with other BQL functions, `univ.holdings()` and `univ.members()` return data as a response that must be converted into a pandas DataFrame for further processing.

---

## Portfolio library

The Portfolio library is a Python module that has been built for programmatic discovery and richer reads. In future releases it will also support programmatic modification, although at present the API is limited to read-only actions.

The Portfolio library can read positions across single or multiple dates for both PORT-E and PORT-D portfolios. It also supports portfolio discovery, with the ability to list available portfolios and their metadata.

In addition to providing a Python-native interface for programmatic access, the Portfolio library also fully interoperates with the pandas data science library.

---

## Feature breakdown

The following table highlights the most important differences between the BQL functions (`univ.holdings()` and `univ.members()`) and the Portfolio library.

| Feature | `univ.holdings()` | `univ.members()` | Portfolio library |
|---|---|---|---|
| **Discovering Portfolios** | No support for portfolio discovery | No support for portfolio discovery | Can access a filterable portfolio catalog |
| **Date Control and History** | Support for any single historical date (limited to PORT-E portfolios) | Only supports most recent date's holdings | Supports single or multiple dates for PORT-E or PORT-D portfolios |
| **Weights and Normalization** | Weights are always normalized to sum to 100 | Weights are always normalized to sum to 100 | Defaults to non-normalized but can be configured to provide normalized weights |
| **Zero Positions** | Must keep zero positions | Must keep zero positions | Defaults to keeping zero positions, but can be configured to drop |
| **Cash Representation** | Always drops PRTU cash and keeps ISO currency | Always drops PRTU cash and keeps ISO currency | Configurable, with shares portfolios treating cash as separate line item with no FIGI by default |

---

## Discovering portfolios

BQL does not offer a "list portfolios" endpoint. Instead, you begin with an identifier you already know.

The Portfolio library supports portfolio lookup by identifier as well, but also offers a **filterable portfolio catalog**. The catalog includes metadata fields useful for access control and workflow routing—for example, distinguishing PORT-E from PORT-D or checking permissions.

---

## Date control and history

The BQL methods provide limited support for retrieving historical portfolio data:

- **`univ.members()`** — Can only return a portfolio's constituents as of the latest date.
- **`univ.holdings()`** — Can retrieve a single historical rebalance date, but only for PORT-E (Enterprise) portfolios.
- Neither function supports retrieving a full history of holdings across many dates.

If you are standardizing a research pipeline or backfilling long windows, the **Portfolio library** is a better fit. By default, it works like the BQL snapshot functions, but can be configured to fetch portfolio data for a single historical date, or even a time series of historical data. Additionally, only the Portfolio library supports PORT-E and PORT-D portfolios.

---

## Weights and normalization

BQL always normalizes weights and ensures they sum to 100.

The Portfolio library uses the opposite default: it returns **unnormalized** positions/weights (as they appear in PRTU). However, the Portfolio library can be configured to return normalized weights in addition to the unnormalized weights.

> **Note:** You may observe slight numerical differences when requesting normalized outputs from both systems. This is expected and typically reflects underlying price source differences in the normalization process rather than a data error.

---

## Zero positions

The BQL functions retain zero-weight securities, which can be useful when you want to preserve the shape of a portfolio across dates even when constituents temporarily drop to zero.

The default behavior of the Portfolio library is also to retain zero-weight securities, but can be configured to drop these securities—for example, when you want a compact, investable snapshot rather than a full audit trail.

---

## Cash representation

BQL removes the explicit "PRTU Cash" line item and instead leaves the ISO currency in the output.

The Portfolio library's default mirrors what you see in PRTU for shares portfolios: cash is a separate line with FIGI = NaN. But as opposed to the BQL functions, the Portfolio library offers customization. If you enable normalized output, PORT cash holdings are folded into the ISO currency representation, aligning the presentation with BQL's.

---

## Getting BQL-like results from the Portfolio library

If you prefer to standardize on the Portfolio library but deliver the same numbers your users expect from BQL, configure it to behave like BQL's defaults. When you call `Portfolio.from_port()`, make sure to:

1. **Use the last available date** (no `dates` parameter), or pass one specific date to mirror `univ.holdings()` behavior.
2. **Turn on normalized weights** by passing `include_normalized_weights=True`. The result will be a normalized list that drops zero-weight lines and presents cash via ISO currency.

---

## See also

- **Portfolio library** — Full documentation for the Portfolio library.
- **univ.holdings()** — BQL documentation for the `univ.holdings()` function.
- **univ.members()** — BQL documentation for the `univ.members()` function.
