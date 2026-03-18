# -*- coding: utf-8 -*-
"""
v2 patch — all improvements:
• Zdog pseudo-3D reactor (boot + header, dragRotate on header)
• Boot voice: "Welcome trader, J.A.R.V.I.S. online"
• Flow Score chart: labels not cut off, bigger brighter font
• Prêmio VOL: shows value in pp (not %)
• Real BBG Greeks: delta_bn, vanna_bn, charm_bn stored in _snapshot['metrics']
• Real Gamma semi-gauge: uses gex_net_bn from metrics
• Regenerates _JARVIS_EXPORT_TEMPLATE + _export_dashboard_html() in dashboard
"""
import sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

TMPL_FILE = r'C:/Users/rafael bentes/Downloads/jarvis_final (1).html'
DASH_FILE  = r'C:/Users/rafael bentes/bbg/examples/greeks_dashboard.py'

# ═══════════════════════════════════════════════════════════════════════════════
# PART A — Modify jarvis_final (1).html
# ═══════════════════════════════════════════════════════════════════════════════
with open(TMPL_FILE, encoding='utf-8') as f:
    tmpl = f.read()

fixes = 0

# ── A1. Add Zdog CDN ─────────────────────────────────────────────────────────
OLD = '</style>\n</head>'
NEW = '</style>\n<script src="https://unpkg.com/zdog@1/dist/zdog.dist.min.js"></script>\n</head>'
if OLD in tmpl:
    tmpl = tmpl.replace(OLD, NEW, 1); fixes += 1; print('A1 Zdog CDN')
else:
    print('WARN A1')

# ── A2. Boot reactor → canvas ────────────────────────────────────────────────
OLD = ('  <div class="brw">\n'
       '    <div class="bri bri1"></div><div class="bri bri2"></div>'
       '<div class="bri bri3"></div>\n'
       '    <div class="bcore"></div>\n'
       '  </div>')
NEW = '  <canvas id="boot-reactor" width="80" height="80" style="display:block;margin-bottom:16px"></canvas>'
if OLD in tmpl:
    tmpl = tmpl.replace(OLD, NEW, 1); fixes += 1; print('A2 boot reactor canvas')
else:
    print('WARN A2')

# ── A3. Header reactor → canvas (dragRotate enabled) ────────────────────────
OLD = ('    <div class="reactor">\n'
       '      <div class="ri ri1"></div><div class="ri ri2"></div><div class="ri ri3"></div>\n'
       '      <div class="rcore"></div>\n'
       '    </div>')
NEW = ('    <canvas id="hdr-reactor" width="32" height="32" '
       'style="flex-shrink:0;cursor:grab"></canvas>')
if OLD in tmpl:
    tmpl = tmpl.replace(OLD, NEW, 1); fixes += 1; print('A3 header reactor canvas')
else:
    print('WARN A3')

# ── A4. Flow chart: taller + brighter rotated labels ────────────────────────
OLD = 'style="flex:1;min-height:280px"><canvas id="flowChart"></canvas>'
NEW = 'style="flex:1;min-height:360px"><canvas id="flowChart"></canvas>'
if OLD in tmpl:
    tmpl = tmpl.replace(OLD, NEW, 1); fixes += 1; print('A4 flowChart height')
else:
    print('WARN A4')

# flowChart x-axis ticks (unique .6 color value)
OLD = "x:{grid:{color:G},ticks:{color:'rgba(0,140,170,.6)'}},"
NEW = ("x:{grid:{color:G},ticks:{color:'rgba(0,200,220,.9)',"
       "font:{size:10,family:\"'Orbitron',sans-serif\"},maxRotation:45,minRotation:45},"
       "border:{color:'rgba(0,80,100,.2)'}},")
if OLD in tmpl:
    tmpl = tmpl.replace(OLD, NEW, 1); fixes += 1; print('A5 flowChart x ticks')
else:
    print('WARN A5')

# flowChart y-axis (also .6)
OLD = "y:{grid:{color:G},ticks:{color:'rgba(0,140,170,.6)'},min:-3.5,max:4,"
NEW = "y:{grid:{color:G},ticks:{color:'rgba(0,180,200,.7)'},min:-3.5,max:4,"
if OLD in tmpl:
    tmpl = tmpl.replace(OLD, NEW, 1); fixes += 1; print('A6 flowChart y ticks')
else:
    print('WARN A6')

# Add layout padding to flowChart options (after maintainAspectRatio:false)
OLD = ("options:{responsive:true,maintainAspectRatio:false,\n"
       "      plugins:{legend:{display:false},tooltip:TT},")
NEW = ("options:{responsive:true,maintainAspectRatio:false,\n"
       "      layout:{padding:{bottom:20}},\n"
       "      plugins:{legend:{display:false},tooltip:TT},")
if OLD in tmpl:
    tmpl = tmpl.replace(OLD, NEW, 1); fixes += 1; print('A7 flowChart padding')
else:
    print('WARN A7')

# ── A5. Prêmio VOL: unit:'%' → state:'pp' ───────────────────────────────────
OLD = "{v:4.76,mn:0,mx:10,label:'PR\u00caMIO VOL',unit:'%',intensity:0.55},"
NEW = "{v:4.76,mn:0,mx:10,label:'PR\u00caMIO VOL',state:'pp',intensity:0.55},"
if OLD in tmpl:
    tmpl = tmpl.replace(OLD, NEW, 1); fixes += 1; print('A8 premioVol pp')
else:
    print('WARN A8')

# ── A6. Gamma semi-gauge: add __JV__ marker ───────────────────────────────────
OLD = "{v:-23.02,mn:-40,mx:0,label:'\u0393 GAMMA (GEX NET)',intensity:0.4},"
NEW = "{v:__JV_V_GEX_SEMI__,mn:-40,mx:40,label:'\u0393 GAMMA (GEX NET)',intensity:0.4},"
if OLD in tmpl:
    tmpl = tmpl.replace(OLD, NEW, 1); fixes += 1; print('A9 Gamma semi marker')
else:
    print('WARN A9')

# ── A7. Delta semi-gauge: add __JV__ markers ─────────────────────────────────
OLD = "{v:-514.42,mn:-800,mx:0,label:'\u0394 DELTA NOCIONAL',intensity:1},"
NEW = "{v:__JV_V_DELTA__,mn:__JV_V_DELTA_MIN__,mx:__JV_V_DELTA_MAX__,label:'\u0394 DELTA NOCIONAL',intensity:1},"
if OLD in tmpl:
    tmpl = tmpl.replace(OLD, NEW, 1); fixes += 1; print('A10 Delta semi marker')
else:
    print('WARN A10')

# ── A8. Vanna semi-gauge: add __JV__ marker ──────────────────────────────────
OLD = "{v:0.03,mn:-1,mx:1,label:'V VANNA',intensity:0.25},"
NEW = "{v:__JV_V_VANNA__,mn:__JV_V_VANNA_MIN__,mx:__JV_V_VANNA_MAX__,label:'V VANNA',intensity:0.25},"
if OLD in tmpl:
    tmpl = tmpl.replace(OLD, NEW, 1); fixes += 1; print('A11 Vanna semi marker')
else:
    print('WARN A11')

# ── A9. Charm semi-gauge: add __JV__ marker ──────────────────────────────────
OLD = "{v:-12.15,mn:-20,mx:0,label:'C CHARM (DI\u00c1RIO)',intensity:0.8},"
NEW = "{v:__JV_V_CHARM__,mn:__JV_V_CHARM_MIN__,mx:__JV_V_CHARM_MAX__,label:'C CHARM (DI\u00c1RIO)',intensity:0.8},"
if OLD in tmpl:
    tmpl = tmpl.replace(OLD, NEW, 1); fixes += 1; print('A12 Charm semi marker')
else:
    print('WARN A12')

# ── A10. Add Zdog reactors + boot voice JS (before // ── BOOT) ───────────────
ZDOG_VOICE_JS = r"""// ── ZDOG REACTORS ──────────────────────────────────────────────────────────
(function(){
  if(typeof Zdog==='undefined') return;
  function mkR(id,sz,drag){
    const cvs=document.getElementById(id); if(!cvs) return;
    cvs.width=sz; cvs.height=sz;
    const illo=new Zdog.Illustration({element:'#'+id,resize:false,zoom:sz/90,dragRotate:!!drag});
    const c1='rgba(0,212,232,.95)',c2='rgba(0,212,232,.55)',c3='rgba(0,212,232,.22)';
    new Zdog.Ellipse({addTo:illo,diameter:62,stroke:3,color:c1,fill:false,rotate:{x:Zdog.TAU/4}});
    new Zdog.Ellipse({addTo:illo,diameter:44,stroke:2.2,color:c2,fill:false,rotate:{x:Zdog.TAU/6,y:Zdog.TAU/8}});
    new Zdog.Ellipse({addTo:illo,diameter:26,stroke:1.5,color:c3,fill:false,rotate:{x:-Zdog.TAU/5,y:-Zdog.TAU/6}});
    new Zdog.Shape({addTo:illo,stroke:10,color:'rgba(0,212,232,1)',translate:{z:5}});
    new Zdog.Shape({addTo:illo,stroke:20,color:'rgba(0,212,232,.1)',translate:{z:3}});
    let t=0;
    (function anim(){t+=0.007;illo.rotate.y=t;illo.updateRenderGraph();requestAnimationFrame(anim)})();
  }
  mkR('boot-reactor',80,false);
  mkR('hdr-reactor',32,true);
})();

// ── BOOT VOICE ────────────────────────────────────────────────────────────────
function _jvSpeak(txt){
  if(!('speechSynthesis' in window)) return;
  speechSynthesis.cancel();
  const u=new SpeechSynthesisUtterance(txt);
  u.lang='en-US'; u.pitch=0.72; u.rate=0.88; u.volume=0.72;
  function go(){speechSynthesis.speak(u);}
  if(speechSynthesis.getVoices().length>0) go();
  else { speechSynthesis.onvoiceschanged=function(){speechSynthesis.onvoiceschanged=null;go();};
         setTimeout(go,250); }
}

"""

OLD_BOOT_MARKER = '// \u2500\u2500 BOOT\n'
if OLD_BOOT_MARKER in tmpl:
    tmpl = tmpl.replace(OLD_BOOT_MARKER, ZDOG_VOICE_JS + OLD_BOOT_MARKER, 1)
    fixes += 1; print('A13 Zdog+voice JS injected')
else:
    print('WARN A13 - trying alternate boot marker')
    ALT = 'const BL=['
    if ALT in tmpl:
        tmpl = tmpl.replace(ALT, ZDOG_VOICE_JS + ALT, 1)
        fixes += 1; print('A13 Zdog+voice via alt marker')

# ── A11. Boot voice call after buildAll() ────────────────────────────────────
OLD_COMPLETE = ("document.getElementById('boot').classList.add('gone');\n"
                "   document.getElementById('app').classList.add('on');buildAll()},500)")
NEW_COMPLETE = ("document.getElementById('boot').classList.add('gone');\n"
                "   document.getElementById('app').classList.add('on');buildAll();"
                "_jvSpeak('Welcome trader. J A R V I S online.')},500)")
if OLD_COMPLETE in tmpl:
    tmpl = tmpl.replace(OLD_COMPLETE, NEW_COMPLETE, 1)
    fixes += 1; print('A14 voice call at boot end')
else:
    print('WARN A14')

print(f'\nHTML fixes: {fixes}')
with open(TMPL_FILE, 'w', encoding='utf-8') as f:
    f.write(tmpl)
print(f'Saved: {TMPL_FILE}')


# ═══════════════════════════════════════════════════════════════════════════════
# PART B — Marker substitutions for export template
# ═══════════════════════════════════════════════════════════════════════════════
subs = [
    # CMD strip
    ('<div class="csv" style="color:rgba(0,212,232,1)">6,716</div>',
     '<div class="csv" style="color:rgba(0,212,232,1)">__JV_SPOT__</div>'),
    ('<div class="csv" style="opacity:.7">6,803</div>',
     '<div class="csv" style="opacity:.7">__JV_FLIP__</div>'),
    ('<div class="csv" style="opacity:.5">\u221223.6B</div>',
     '<div class="csv" style="opacity:.5">__JV_GEX__</div>'),
    ('<div class="csv" style="opacity:.7">1.50\u00d7</div>',
     '<div class="csv" style="opacity:.7">__JV_PC__</div>'),
    ('<div class="csv" style="opacity:.9">+4.8pp</div>',
     '<div class="csv" style="opacity:.9">__JV_IVRV__</div>'),
    ('<div class="csv">76/100</div>',
     '<div class="csv">__JV_SQ__</div>'),
    ('<div class="csv" style="opacity:.7">50.5/100</div>',
     '<div class="csv" style="opacity:.7">__JV_TAIL__</div>'),
    # PAINEL vol cards
    ('<div class="cdv">18.36%</div>',
     '<div class="cdv">__JV_IV30__</div>'),
    ('<div class="cdv" style="opacity:.7">13.60%</div>',
     '<div class="cdv" style="opacity:.7">__JV_RV30__</div>'),
    ('<div class="cdv">+4.76%</div>',
     '<div class="cdv">__JV_IVRV_PREM__</div>'),
    # Key levels
    ('<div class="cdv">~6,803</div>',
     '<div class="cdv">~__JV_FLIP__</div>'),
    ('<div class="cdv">6,800</div>',
     '<div class="cdv">__JV_CW__</div>'),
    ('<div class="cdv" style="opacity:.55">6,700</div>',
     '<div class="cdv" style="opacity:.55">__JV_PW__</div>'),
    # RISCO: tail risk header
    ('<div class="ph"><div class="phd"></div>TAIL RISK \u2014 50/100</div>',
     '<div class="ph"><div class="phd"></div>TAIL RISK \u2014 __JV_TAIL_INT__/100</div>'),
    ('>50.5 / 100</span>',
     '>__JV_TAIL_NUM__ / 100</span>'),
    ('"sbf" style="width:50.5%"',
     '"sbf" style="width:__JV_TAIL_PCT__%"'),
    # RISCO: squeeze header
    ('<div class="ph"><div class="phd" style="animation:blink 1s infinite"></div>GAMMA SQUEEZE \u2014 75.8/100</div>',
     '<div class="ph"><div class="phd" style="animation:blink 1s infinite"></div>GAMMA SQUEEZE \u2014 __JV_SQ_NUM__/100</div>'),
    ('>75.8 / 100</span>',
     '>__JV_SQ_NUM__ / 100</span>'),
    ('"sbf" style="width:75.8%"',
     '"sbf" style="width:__JV_SQ_PCT__%"'),
    # ESTRUTURA subtitle
    ('Spot 6,716 | G-Flip 6,803',
     'Spot __JV_SPOT__ | G-Flip __JV_FLIP__'),
    # buildAll() arc gauges
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
    # GEX chart JS
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
    # Ticker JS
    ("['SPX','6,716',1]",
     "['SPX','__JV_SPOT__',1]"),
    ("['GEX','\u2212$23.6B',0]",
     "['GEX','__JV_GEX_T__',0]"),
    ("['GAMMA FLIP','6,803',0]",
     "['GAMMA FLIP','__JV_FLIP__',0]"),
    ("['IV 30D','18.36%',0]",
     "['IV 30D','__JV_IV30__',0]"),
    ("['RV 30D','13.60%',1]",
     "['RV 30D','__JV_RV30__',1]"),
    ("['P/C','1.50\u00d7',0]",
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
        tmpl = tmpl.replace(old, new); ok_subs += 1
    else:
        print(f'  WARN sub: {repr(old[:60])}')
print(f'Marker subs: {ok_subs}/{len(subs)}')

assert '"""' not in tmpl, 'Triple quotes in template!'


# ═══════════════════════════════════════════════════════════════════════════════
# PART C — New _export_dashboard_html() function
# ═══════════════════════════════════════════════════════════════════════════════
NEW_FUNC = r'''def _export_dashboard_html():
    """Exporta JARVIS HUD HTML standalone — 100% fiel ao jarvis_final v2."""
    if not _snapshot.get('ts'):
        return None

    m    = _snapshot.get('metrics', {})
    spot = _snapshot['spot']

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _f(k, default=0):
        v = m.get(k, default)
        return float(v) if isinstance(v, (int, float)) else default

    def _sym_range(v, factor=1.5, min_abs=1.0):
        """Symmetric range centred on 0 for semi-gauges."""
        mag = max(min_abs, abs(v) * factor)
        return round(-mag, 1), round(mag, 1)

    # ── Metric strings ────────────────────────────────────────────────────────
    _spot_s      = f"{spot:,.0f}"
    _flip_raw    = _f('gamma_flip')
    _flip_s      = f"{_flip_raw:,.0f}" if _flip_raw else "N/A"
    _flip_num    = round(_flip_raw)
    _spot_r10    = round(spot / 10) * 10
    _flip_r10    = round(_flip_num / 10) * 10

    _gex_raw     = _f('gex_net_bn')
    _gex_sign    = "\u2212" if _gex_raw < 0 else "+"
    _gex_s       = f"{_gex_sign}{abs(_gex_raw):.1f}B"
    _gex_t       = f"{_gex_sign}${abs(_gex_raw):.1f}B"

    _pc_raw      = _f('pc_ratio')
    _pc_s        = f"{_pc_raw:.2f}\u00d7"

    _ivrv_raw    = _f('iv_rv_pp')
    _ivrv_s      = f"{_ivrv_raw:+.1f}pp"
    _ivrv_prem_s = f"{_ivrv_raw:+.2f}%"

    _sq_raw      = _f('squeeze_score')
    _sq_s        = f"{_sq_raw:.0f}/100"
    _sq_int_s    = f"{_sq_raw:.0f}"

    _tail_raw    = _f('tail_score')
    _tail_s      = f"{_tail_raw:.1f}/100"
    _tail_num_s  = f"{_tail_raw:.1f}"
    _tail_int_s  = f"{_tail_raw:.0f}"

    _iv30_raw    = _f('iv_30d')
    _rv30_raw    = _f('rv_30d')
    _iv30_s      = f"{_iv30_raw*100:.2f}%"
    _rv30_s      = f"{_rv30_raw*100:.2f}%"

    _cw_raw      = _f('call_wall')
    _pw_raw      = _f('put_wall')
    _cw_s        = f"{_cw_raw:,.0f}" if _cw_raw else "N/A"
    _pw_s        = f"{_pw_raw:,.0f}" if _pw_raw else "N/A"

    # ── JS gauge values ───────────────────────────────────────────────────────
    _frag_raw    = _f('fragility')
    _frag_v      = round(_frag_raw * 100, 2) if abs(_frag_raw) <= 1.0 else round(abs(_frag_raw), 2)
    _move_raw    = _f('daily_move')
    _move_v      = round(abs(_move_raw) * 100, 2) if abs(_move_raw) <= 1.0 else round(abs(_move_raw), 2)
    _ivrv_v      = round(abs(_ivrv_raw), 2)
    _sq_v        = round(_sq_raw, 1)
    _tail_v      = round(_tail_raw, 1)

    # ── Greek semi-gauge values (real BBG) ────────────────────────────────────
    _delta_v     = round(_f('delta_bn'), 2)
    _delta_min, _delta_max = _sym_range(_delta_v, factor=1.5, min_abs=5.0)
    _vanna_v     = round(_f('vanna_bn'), 3)
    _vanna_min, _vanna_max = _sym_range(_vanna_v, factor=2.0, min_abs=0.5)
    _charm_v     = round(_f('charm_bn'), 3)
    _charm_min, _charm_max = _sym_range(_charm_v, factor=2.0, min_abs=0.2)
    _gex_semi    = round(_gex_raw, 2)

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
    # Greek semi-gauges (real BBG)
    _html = _html.replace('__JV_V_GEX_SEMI__',  str(_gex_semi))
    _html = _html.replace('__JV_V_DELTA__',      str(_delta_v))
    _html = _html.replace('__JV_V_DELTA_MIN__',  str(_delta_min))
    _html = _html.replace('__JV_V_DELTA_MAX__',  str(_delta_max))
    _html = _html.replace('__JV_V_VANNA__',      str(_vanna_v))
    _html = _html.replace('__JV_V_VANNA_MIN__',  str(_vanna_min))
    _html = _html.replace('__JV_V_VANNA_MAX__',  str(_vanna_max))
    _html = _html.replace('__JV_V_CHARM__',      str(_charm_v))
    _html = _html.replace('__JV_V_CHARM_MIN__',  str(_charm_min))
    _html = _html.replace('__JV_V_CHARM_MAX__',  str(_charm_max))

    return _html
'''


# ═══════════════════════════════════════════════════════════════════════════════
# PART D — Patch greeks_dashboard.py
# ═══════════════════════════════════════════════════════════════════════════════
with open(DASH_FILE, 'r', encoding='utf-8') as f:
    c = f.read()

# Find old block start (template constant or function def)
for start_marker in [
    '_JARVIS_EXPORT_TEMPLATE = """',
    'def _export_dashboard_html():\n    """Exporta JARVIS HUD HTML standalone',
    'def _export_dashboard_html():\n    """Exporta como JARVIS HUD HTML standalone',
]:
    idx = c.find(start_marker)
    if idx != -1:
        break

if idx == -1:
    print('ERROR: cannot find export function in dashboard')
    sys.exit(1)

# Find end: last `    return _html` or `    return html`
end_pattern = re.compile(r'\n    return (?:_html|html)\n')
match = None
for m2 in end_pattern.finditer(c, idx):
    match = m2
if not match:
    print('ERROR: cannot find function end')
    sys.exit(1)
end_idx = match.end()

TEMPLATE_BLOCK = '_JARVIS_EXPORT_TEMPLATE = """\n' + tmpl + '\n"""\n\n'
NEW_CODE = TEMPLATE_BLOCK + NEW_FUNC + '\n'

print(f'Replacing chars {idx}..{end_idx} in dashboard')
c = c[:idx] + NEW_CODE + c[end_idx:]

with open(DASH_FILE, 'w', encoding='utf-8') as f:
    f.write(c)
print('Dashboard patched')

# ── Also add delta_bn/vanna_bn/charm_bn to _snapshot['metrics'] ─────────────
with open(DASH_FILE, 'r', encoding='utf-8') as f:
    c2 = f.read()

OLD_METRICS_END = (
    "                'fragility':     fragility  if 'fragility'  in dir() else 0,\n"
    "            }"
)
NEW_METRICS_END = (
    "                'fragility':     fragility  if 'fragility'  in dir() else 0,\n"
    "                'delta_bn':      delta_bn   if 'delta_bn'   in dir() else 0,\n"
    "                'vanna_bn':      vanna_bn   if 'vanna_bn'   in dir() else 0,\n"
    "                'charm_bn':      charm_bn   if 'charm_bn'   in dir() else 0,\n"
    "            }"
)
if OLD_METRICS_END in c2:
    c2 = c2.replace(OLD_METRICS_END, NEW_METRICS_END, 1)
    with open(DASH_FILE, 'w', encoding='utf-8') as f:
        f.write(c2)
    print('Greek metrics added to _snapshot')
else:
    print('WARN: metrics end block not found — Greeks not added to snapshot')

# ── Syntax check ─────────────────────────────────────────────────────────────
import ast
with open(DASH_FILE, encoding='utf-8') as f:
    src = f.read()
try:
    ast.parse(src)
    print('Syntax: OK')
except SyntaxError as e:
    print(f'Syntax ERROR: {e}')

print('\nDone.')
