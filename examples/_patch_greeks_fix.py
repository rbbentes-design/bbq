# -*- coding: utf-8 -*-
"""
Fix:
1. _greek_cache — store delta_bn/10, vanna_bn, charm_bn/10 inside build_greek_overview()
2. _snapshot['metrics'] — use cache + add fp_score z-components
3. JARVIS flow chart — 8 real components via __JV_FLOW_DATA__ marker
4. _export_dashboard_html() — inject flow data
"""
import sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
DASH = r'C:/Users/rafael bentes/bbg/examples/greeks_dashboard.py'

with open(DASH, encoding='utf-8') as f:
    c = f.read()

ok = 0

# ── 1. Add _greek_cache initializer next to _snapshot init ──────────────────
OLD_SNAP_INIT = "_snapshot = {'sections': [], 'ticker': '', 'spot': 0, 'ts': '', 'metrics': {}}"
NEW_SNAP_INIT = ("_greek_cache = {}  # populated by build_greek_overview()\n"
                 "_snapshot = {'sections': [], 'ticker': '', 'spot': 0, 'ts': '', 'metrics': {}}")
if OLD_SNAP_INIT in c:
    c = c.replace(OLD_SNAP_INIT, NEW_SNAP_INIT, 1); ok += 1
    print('1. _greek_cache init added')
else:
    print('WARN 1: snapshot init not found')

# ── 2. Store Greeks in cache inside build_greek_overview() after charm_bn ────
OLD_CHARM_LINE = (
    "    charm_bn = float(np.nansum(greeks_now['charm'] * oi_100) * spot / 365.0 / 1e9)\n"
    "\n"
    "    # Escala din\u00e2mica m\u00ednima por grega (SPX t\u00edpico)"
)
NEW_CHARM_LINE = (
    "    charm_bn = float(np.nansum(greeks_now['charm'] * oi_100) * spot / 365.0 / 1e9)\n"
    "\n"
    "    # Cache module-level para exporta\u00e7\u00e3o JARVIS (div/10 = escala BBG)\n"
    "    _greek_cache['delta_bn'] = delta_bn / 10\n"
    "    _greek_cache['vanna_bn'] = vanna_bn\n"
    "    _greek_cache['charm_bn'] = charm_bn / 10\n"
    "\n"
    "    # Escala din\u00e2mica m\u00ednima por grega (SPX t\u00edpico)"
)
if OLD_CHARM_LINE in c:
    c = c.replace(OLD_CHARM_LINE, NEW_CHARM_LINE, 1); ok += 1
    print('2. Greek cache populated in build_greek_overview()')
else:
    print('WARN 2: charm_bn line not found')

# ── 3. Fix snapshot metrics: use _greek_cache + add flow score z-components ──
OLD_METRICS = (
    "                'delta_bn':      delta_bn   if 'delta_bn'   in dir() else 0,\n"
    "                'vanna_bn':      vanna_bn   if 'vanna_bn'   in dir() else 0,\n"
    "                'charm_bn':      charm_bn   if 'charm_bn'   in dir() else 0,\n"
    "            }"
)
NEW_METRICS = (
    "                'delta_bn':      _greek_cache.get('delta_bn', 0),\n"
    "                'vanna_bn':      _greek_cache.get('vanna_bn', 0),\n"
    "                'charm_bn':      _greek_cache.get('charm_bn', 0),\n"
    "                # Flow score z-components (real BBG)\n"
    "                'z_cta':         fp_score.get('z_cta', 0)        if isinstance(fp_score, dict) else 0,\n"
    "                'z_dealer':      fp_score.get('z_dealer', 0)     if isinstance(fp_score, dict) else 0,\n"
    "                'z_volctrl':     fp_score.get('z_volctrl', 0)    if isinstance(fp_score, dict) else 0,\n"
    "                'z_rp':          fp_score.get('z_rp', 0)         if isinstance(fp_score, dict) else 0,\n"
    "                'z_leveraged':   fp_score.get('z_leveraged', 0)  if isinstance(fp_score, dict) else 0,\n"
    "                'z_passive_etf': fp_score.get('z_passive_etf', 0) if isinstance(fp_score, dict) else 0,\n"
    "                'z_buyback':     fp_score.get('z_buyback', 0)    if isinstance(fp_score, dict) else 0,\n"
    "                'z_cot':         fp_score.get('z_cot', 0)        if isinstance(fp_score, dict) else 0,\n"
    "            }"
)
if OLD_METRICS in c:
    c = c.replace(OLD_METRICS, NEW_METRICS, 1); ok += 1
    print('3. Snapshot metrics: Greeks + flow z-scores added')
else:
    print('WARN 3: metrics block not found')

# ── 4. Update flow chart in _JARVIS_EXPORT_TEMPLATE: 5→8 components ─────────
# Replace the hardcoded 5-label chart with 8 labels + __JV_FLOW_DATA__ marker
OLD_FLOW_DATA = (
    "      labels:['CTA','Dealer/MM','Vol Ctrl','Risk Parity','ETFs Alav.'],\n"
    "      datasets:[{\n"
    "        label:'Z-Score',\n"
    "        data:[-2.1,0.0,3.0,3.0,0.37],"
)
NEW_FLOW_DATA = (
    "      labels:['CTA','Dealer/MM','Vol Ctrl','Risk Parity','ETFs Alav.','ETFs Pass.','Buyback','COT'],\n"
    "      datasets:[{\n"
    "        label:'Z-Score',\n"
    "        data:[__JV_FLOW_DATA__],"
)
if OLD_FLOW_DATA in c:
    c = c.replace(OLD_FLOW_DATA, NEW_FLOW_DATA, 1); ok += 1
    print('4. Flow chart: 8 components + __JV_FLOW_DATA__ marker')
else:
    print('WARN 4: flow chart data not found')

# ── 5. Add __JV_FLOW_DATA__ replacement to _export_dashboard_html() ──────────
OLD_EXPORT_END = (
    "    _html = _html.replace('__JV_V_CHARM_MAX__',  str(_charm_max))\n"
    "\n"
    "    return _html"
)
NEW_EXPORT_END = (
    "    _html = _html.replace('__JV_V_CHARM_MAX__',  str(_charm_max))\n"
    "    # Flow score — 8 real BBG components\n"
    "    import json as _json\n"
    "    _flow_data = _json.dumps([\n"
    "        round(_f('z_cta'), 2),\n"
    "        round(_f('z_dealer'), 2),\n"
    "        round(_f('z_volctrl'), 2),\n"
    "        round(_f('z_rp'), 2),\n"
    "        round(_f('z_leveraged'), 2),\n"
    "        round(_f('z_passive_etf'), 2),\n"
    "        round(_f('z_buyback'), 2),\n"
    "        round(_f('z_cot'), 2),\n"
    "    ])\n"
    "    _html = _html.replace('[__JV_FLOW_DATA__]', _flow_data)\n"
    "\n"
    "    return _html"
)
if OLD_EXPORT_END in c:
    c = c.replace(OLD_EXPORT_END, NEW_EXPORT_END, 1); ok += 1
    print('5. Flow data injection added to export function')
else:
    print('WARN 5: export function end not found')

# ── Write + syntax check ─────────────────────────────────────────────────────
with open(DASH, 'w', encoding='utf-8') as f:
    f.write(c)

import ast
try:
    ast.parse(c)
    print(f'\nSyntax: OK  ({ok}/5 changes)')
except SyntaxError as e:
    print(f'Syntax ERROR: {e}')
