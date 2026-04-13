"""
MARKET MAKER DASHBOARD — Versao Modular v2.0
Refatoracao do greeks_dashboard.py monolitico em modulos independentes.

Modulos:
    config          Imports, constantes, CSS, design system
    ui              HUD panels, SVG gauges, badges
    greeks          Motor Black-Scholes (vetorial)
    data            Pipeline BQL (fetch market data, options chain, historical)
    exposure        Exposicoes por strike, curvas modelo, walls, risk models
    flows           ETF rebalancing, COT, buyback, flow scoring, CTA, dealer, vol control
    flow_charts     Visualizacoes do Flow Patrol
    dispersion      Analise de dispersao: correlacao, pairs, straddles, ML
    charts          Skew, gamma, squeeze, vol smile, dynamic book, decision engine
    dashboard       Orquestrador: overview, callback principal, widgets

Uso no BQuant:
    from greeks_dashboard_separado.dashboard import *
"""
