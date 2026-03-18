# -*- coding: utf-8 -*-
"""
Replace _export_dashboard_html() with a version that uses jarvis_final template.
Embeds _JARVIS_EXPORT_TEMPLATE as a module-level constant + new slim function.
"""
import sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

TMPL_FILE = r'C:/Users/rafael bentes/Downloads/jarvis_final (1).html'
DASH_FILE = r'C:/Users/rafael bentes/bbg/examples/greeks_dashboard.py'

# ─── 1. Read template and apply __JV_XXX__ markers ────────────────────────────
with open(TMPL_FILE, encoding='utf-8') as f:
    tmpl = f.read()

subs = [
    # ── CMD strip ──────────────────────────────────────────────────────────────
    ('<div class="csv" style="color:rgba(0,212,232,1)">6,716</div>',
     '<div class="csv" style="color:rgba(0,212,232,1)">__JV_SPOT__</div>'),
    ('<div class="csv" style="opacity:.7">6,803</div>',      # GAMMA FLIP
     '<div class="csv" style="opacity:.7">__JV_FLIP__</div>'),
    ('<div class="csv" style="opacity:.5">\u221223.6B</div>',  # GEX (−23.6B)
     '<div class="csv" style="opacity:.5">__JV_GEX__</div>'),
    ('<div class="csv" style="opacity:.7">1.50\u00d7</div>',  # P/C (×)
     '<div class="csv" style="opacity:.7">__JV_PC__</div>'),
    ('<div class="csv" style="opacity:.9">+4.8pp</div>',
     '<div class="csv" style="opacity:.9">__JV_IVRV__</div>'),
    ('<div class="csv">76/100</div>',                         # SQUEEZE
     '<div class="csv">__JV_SQ__</div>'),
    ('<div class="csv" style="opacity:.7">50.5/100</div>',    # TAIL
     '<div class="csv" style="opacity:.7">__JV_TAIL__</div>'),

    # ── PAINEL vol cards ───────────────────────────────────────────────────────
    ('<div class="cdv">18.36%</div>',
     '<div class="cdv">__JV_IV30__</div>'),
    ('<div class="cdv" style="opacity:.7">13.60%</div>',
     '<div class="cdv" style="opacity:.7">__JV_RV30__</div>'),
    ('<div class="cdv">+4.76%</div>',
     '<div class="cdv">__JV_IVRV_PREM__</div>'),

    # ── PAINEL key levels ──────────────────────────────────────────────────────
    ('<div class="cdv">~6,803</div>',
     '<div class="cdv">~__JV_FLIP__</div>'),
    ('<div class="cdv">6,800</div>',
     '<div class="cdv">__JV_CW__</div>'),
    ('<div class="cdv" style="opacity:.55">6,700</div>',
     '<div class="cdv" style="opacity:.55">__JV_PW__</div>'),

    # ── RISCO: tail risk panel header ──────────────────────────────────────────
    ('<div class="ph"><div class="phd"></div>TAIL RISK \u2014 50/100</div>',
     '<div class="ph"><div class="phd"></div>TAIL RISK \u2014 __JV_TAIL_INT__/100</div>'),
    # score bar value text
    ('>50.5 / 100</span>',
     '>__JV_TAIL_NUM__ / 100</span>'),
    # score bar width
    ('"sbf" style="width:50.5%"',
     '"sbf" style="width:__JV_TAIL_PCT__%"'),

    # ── RISCO: squeeze panel header ────────────────────────────────────────────
    ('<div class="ph"><div class="phd" style="animation:blink 1s infinite"></div>GAMMA SQUEEZE \u2014 75.8/100</div>',
     '<div class="ph"><div class="phd" style="animation:blink 1s infinite"></div>GAMMA SQUEEZE \u2014 __JV_SQ_NUM__/100</div>'),
    ('>75.8 / 100</span>',
     '>__JV_SQ_NUM__ / 100</span>'),
    ('"sbf" style="width:75.8%"',
     '"sbf" style="width:__JV_SQ_PCT__%"'),

    # ── ESTRUTURA GEX subtitle ─────────────────────────────────────────────────
    ('Spot 6,716 | G-Flip 6,803',
     'Spot __JV_SPOT__ | G-Flip __JV_FLIP__'),

    # ── buildAll() JS: arc gauge values ───────────────────────────────────────
    ('{v:8.43,mn:0,mx:20,',
     '{v:__JV_V_FRAG__,mn:0,mx:20,'),
    ('{v:4.76,mn:0,mx:10,',
     '{v:__JV_V_IVRV__,mn:0,mx:10,'),
    ('{v:1.16,mn:0,mx:5,',
     '{v:__JV_V_MOVE__,mn:0,mx:5,'),
    ('{v:50.5,mn:0,mx:100,',
     '{v:__JV_V_TAIL__,mn:0,mx:100,'),
    ('{v:75.8,mn:0,mx:100,',
     '{v:__JV_V_SQ__,mn:0,mx:100,'),

    # ── buildAll() JS: GEX chart ───────────────────────────────────────────────
    ('const x=(s-6803)/200;',
     'const x=(s-__JV_FLIP_NUM__)/200;'),
    ('Math.abs(s-6720)<6',
     'Math.abs(s-__JV_SPOT_R10__)<6'),
    ("data:strikes.map((s,i)=>Math.abs(s-6800)<6?gex[i]:null)",
     "data:strikes.map((s,i)=>Math.abs(s-__JV_FLIP_R10__)<6?gex[i]:null)"),
    ("{label:'Spot 6,716'",
     "{label:'Spot __JV_SPOT__'"),
    ("{label:'G-Flip 6,803'",
     "{label:'G-Flip __JV_FLIP__'"),

    # ── Ticker JS array ────────────────────────────────────────────────────────
    ("['SPX','6,716',1]",
     "['SPX','__JV_SPOT__',1]"),
    ("['\u2212$23.6B',0]",           # −$23.6B (unicode minus)
     "['__JV_GEX_T__',0]"),
    ("['GAMMA FLIP','6,803',0]",
     "['GAMMA FLIP','__JV_FLIP__',0]"),
    ("['IV 30D','18.36%',0]",
     "['IV 30D','__JV_IV30__',0]"),
    ("['RV 30D','13.60%',1]",
     "['RV 30D','__JV_RV30__',1]"),
    ("['P/C','1.50\u00d7',0]",       # × char
     "['P/C','__JV_PC_T__',0]"),
    ("['PUT WALL','6,700',0]",
     "['PUT WALL','__JV_PW__',0]"),
    ("['CALL WALL','6,800',1]",
     "['CALL WALL','__JV_CW__',1]"),
    ("['TAIL RISK','50.5',0]",
     "['TAIL RISK','__JV_TAIL_NUM__',0]"),
    ("['SQUEEZE','76/100',0]",
     "['SQUEEZE','__JV_SQ_T__',0]"),
]

ok_subs = 0
for old, new in subs:
    if old in tmpl:
        tmpl = tmpl.replace(old, new)
        ok_subs += 1
    else:
        print(f"WARNING not found: {repr(old[:70])}")

print(f"Template markers applied: {ok_subs}/{len(subs)}")

# Safety check: no triple-double-quotes in template
assert '"""' not in tmpl, "ERROR: triple quotes found in template!"

# ─── 2. Build the Python code that replaces the function ─────────────────────
NEW_CODE = (
    '_JARVIS_EXPORT_TEMPLATE = """\n'
    + tmpl
    + '\n"""\n\n'
    + r'''def _export_dashboard_html():
    """Exporta JARVIS HUD HTML standalone — 100% fiel ao jarvis_final design."""
    if not _snapshot.get('ts'):
        return None

    m = _snapshot.get('metrics', {})
    spot = _snapshot['spot']
    import math as _math

    # ── Compute display values ────────────────────────────────────────────────
    _spot_s      = f"{spot:,.0f}"
    _flip_raw    = float(m.get('gamma_flip', 0) or 0)
    _flip_s      = f"{_flip_raw:,.0f}" if _flip_raw else "N/A"
    _flip_num    = round(_flip_raw)
    _spot_r10    = round(spot / 10) * 10
    _flip_r10    = round(_flip_num / 10) * 10

    _gex_raw     = float(m.get('gex_net_bn', 0) or 0)
    _gex_sign    = "\u2212" if _gex_raw < 0 else "+"
    _gex_s       = f"{_gex_sign}{abs(_gex_raw):.1f}B"
    _gex_t       = f"{_gex_sign}${abs(_gex_raw):.1f}B"

    _pc_raw      = float(m.get('pc_ratio', 0) or 0)
    _pc_s        = f"{_pc_raw:.2f}\u00d7"

    _ivrv_raw    = float(m.get('iv_rv_pp', 0) or 0)
    _ivrv_s      = f"{_ivrv_raw:+.1f}pp"
    _ivrv_prem_s = f"{_ivrv_raw:+.2f}%"

    _sq_raw      = m.get('squeeze_score', 0)
    _sq_num      = float(_sq_raw) if isinstance(_sq_raw, (int, float)) else 0.0
    _sq_s        = f"{_sq_num:.0f}/100"
    _sq_int_s    = f"{_sq_num:.0f}"

    _tail_raw    = float(m.get('tail_score', 0) or 0)
    _tail_s      = f"{_tail_raw:.1f}/100"
    _tail_num_s  = f"{_tail_raw:.1f}"
    _tail_int_s  = f"{_tail_raw:.0f}"

    _iv30_raw    = float(m.get('iv_30d', 0) or 0)
    _rv30_raw    = float(m.get('rv_30d', 0) or 0)
    _iv30_s      = f"{_iv30_raw*100:.2f}%"
    _rv30_s      = f"{_rv30_raw*100:.2f}%"

    _cw_raw      = float(m.get('call_wall', 0) or 0)
    _pw_raw      = float(m.get('put_wall',  0) or 0)
    _cw_s        = f"{_cw_raw:,.0f}" if _cw_raw else "N/A"
    _pw_s        = f"{_pw_raw:,.0f}" if _pw_raw else "N/A"

    _frag_raw    = float(m.get('fragility', 0) or 0)
    _frag_v      = round(_frag_raw * 100, 2) if abs(_frag_raw) <= 1.0 else round(abs(_frag_raw), 2)
    _move_raw    = float(m.get('daily_move', 0) or 0)
    _move_v      = round(abs(_move_raw) * 100, 2) if abs(_move_raw) <= 1.0 else round(abs(_move_raw), 2)
    _ivrv_v      = round(abs(_ivrv_raw), 2)
    _sq_v        = round(_sq_num, 1)
    _tail_v      = round(_tail_raw, 1)

    # ── Apply replacements ────────────────────────────────────────────────────
    _html = _JARVIS_EXPORT_TEMPLATE
    _html = _html.replace('__JV_SPOT__',       _spot_s)
    _html = _html.replace('__JV_FLIP__',       _flip_s)
    _html = _html.replace('__JV_GEX__',        _gex_s)
    _html = _html.replace('__JV_PC__',         _pc_s)
    _html = _html.replace('__JV_IVRV__',       _ivrv_s)
    _html = _html.replace('__JV_SQ__',         _sq_s)
    _html = _html.replace('__JV_TAIL__',       _tail_s)
    _html = _html.replace('__JV_IV30__',       _iv30_s)
    _html = _html.replace('__JV_RV30__',       _rv30_s)
    _html = _html.replace('__JV_IVRV_PREM__',  _ivrv_prem_s)
    _html = _html.replace('__JV_CW__',         _cw_s)
    _html = _html.replace('__JV_PW__',         _pw_s)
    _html = _html.replace('__JV_TAIL_INT__',   _tail_int_s)
    _html = _html.replace('__JV_TAIL_NUM__',   _tail_num_s)
    _html = _html.replace('__JV_TAIL_PCT__',   _tail_num_s)
    _html = _html.replace('__JV_SQ_NUM__',     _sq_int_s)
    _html = _html.replace('__JV_SQ_PCT__',     _sq_int_s)
    _html = _html.replace('__JV_V_FRAG__',     str(_frag_v))
    _html = _html.replace('__JV_V_IVRV__',     str(_ivrv_v))
    _html = _html.replace('__JV_V_MOVE__',     str(_move_v))
    _html = _html.replace('__JV_V_TAIL__',     str(_tail_v))
    _html = _html.replace('__JV_V_SQ__',       str(_sq_v))
    _html = _html.replace('__JV_FLIP_NUM__',   str(_flip_num))
    _html = _html.replace('__JV_SPOT_R10__',   str(_spot_r10))
    _html = _html.replace('__JV_FLIP_R10__',   str(_flip_r10))
    _html = _html.replace('__JV_GEX_T__',      _gex_t)
    _html = _html.replace('__JV_PC_T__',       _pc_s)
    _html = _html.replace('__JV_SQ_T__',       _sq_s)

    return _html'''
    + '\n'
)

# ─── 3. Patch greeks_dashboard.py ────────────────────────────────────────────
with open(DASH_FILE, 'r', encoding='utf-8') as f:
    c = f.read()

# Find old export function boundaries
OLD_FUNC_START = '_JARVIS_EXPORT_TEMPLATE = """'
OLD_FUNC_START2 = 'def _export_dashboard_html():\n    """Exporta JARVIS HUD HTML standalone'

# Check which version is in the file
if OLD_FUNC_START in c:
    # Already has _JARVIS_EXPORT_TEMPLATE — find from that to end of function
    start_idx = c.find(OLD_FUNC_START)
elif OLD_FUNC_START2 in c:
    # Has the previous version (no template constant)
    start_idx = c.find(OLD_FUNC_START2)
else:
    # Older version
    OLD_FUNC_START3 = 'def _export_dashboard_html():\n    """Exporta como JARVIS HUD HTML standalone'
    start_idx = c.find(OLD_FUNC_START3)

if start_idx == -1:
    print("ERROR: could not find function start")
    sys.exit(1)

# Find end: the "return _html" or "return html" at 4-space indent that closes the function
# Look for the last such occurrence after start_idx
end_pattern = re.compile(r'\n    return (?:_html|html)\n')
match = None
for m2 in end_pattern.finditer(c, start_idx):
    match = m2
if not match:
    print("ERROR: could not find function end")
    sys.exit(1)

end_idx = match.end()

print(f"Replacing chars {start_idx}..{end_idx} (len={end_idx-start_idx})")
c = c[:start_idx] + NEW_CODE + c[end_idx:]

with open(DASH_FILE, 'w', encoding='utf-8') as f:
    f.write(c)
print("greeks_dashboard.py patched OK")

# ─── 4. Quick syntax check ────────────────────────────────────────────────────
import ast
try:
    with open(DASH_FILE, encoding='utf-8') as f:
        ast.parse(f.read())
    print("Syntax: OK")
except SyntaxError as e:
    print(f"Syntax ERROR: {e}")
