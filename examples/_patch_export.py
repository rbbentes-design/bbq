# -*- coding: utf-8 -*-
"""Replace _export_dashboard_html() with JARVIS HUD export + add metrics to _snapshot."""

with open(r'C:/Users/rafael bentes/bbg/examples/greeks_dashboard.py', 'r', encoding='utf-8') as f:
    c = f.read()

ok = 0

# ═══════════════════════════════════════════════════════════════════════════
# 1. ADD metrics dict to _snapshot initializer (line ~581)
# ═══════════════════════════════════════════════════════════════════════════
OLD_SNAP_INIT = "_snapshot = {'sections': [], 'ticker': '', 'spot': 0, 'ts': ''}"
NEW_SNAP_INIT = "_snapshot = {'sections': [], 'ticker': '', 'spot': 0, 'ts': '', 'metrics': {}}"
if OLD_SNAP_INIT in c:
    c = c.replace(OLD_SNAP_INIT, NEW_SNAP_INIT, 1); ok += 1
    print("snapshot init updated")
else:
    print("NOT FOUND: snapshot init")

# ═══════════════════════════════════════════════════════════════════════════
# 2. POPULATE _snapshot['metrics'] right after spot/ts are set (~line 10549)
# ═══════════════════════════════════════════════════════════════════════════
OLD_SNAP_POP = (
    "            _snapshot['ticker'] = ticker\n"
    "            _snapshot['spot'] = spot\n"
    "            _snapshot['ts'] = datetime.now().strftime('%Y-%m-%d %H:%M')\n"
    "            _snapshot['sections'] = []"
)
NEW_SNAP_POP = (
    "            _snapshot['ticker'] = ticker\n"
    "            _snapshot['spot'] = spot\n"
    "            _snapshot['ts'] = datetime.now().strftime('%Y-%m-%d %H:%M')\n"
    "            _snapshot['sections'] = []\n"
    "            _snapshot['metrics'] = {\n"
    "                'gamma_flip':    gamma_flip,\n"
    "                'gex_net_bn':    (_sq_gex_v1 * 0.1) if '_sq_gex_v1' in dir() else 0,\n"
    "                'pc_ratio':      _sq_pc_v1  if '_sq_pc_v1'  in dir() else 0,\n"
    "                'iv_rv_pp':      (iv_30d - rv_30d) * 100 if pd.notna(iv_30d) and pd.notna(rv_30d) else 0,\n"
    "                'iv_30d':        iv_30d if pd.notna(iv_30d) else 0,\n"
    "                'rv_30d':        rv_30d if pd.notna(rv_30d) else 0,\n"
    "                'squeeze_score': _sq_score_disp if '_sq_score_disp' in dir() else 'N/A',\n"
    "                'tail_score':    analytics.get('tail_score', 0) if analytics else 0,\n"
    "                'call_wall':     call_wall,\n"
    "                'put_wall':      put_wall,\n"
    "                'daily_move':    daily_move if 'daily_move' in dir() else 0,\n"
    "                'fragility':     fragility  if 'fragility'  in dir() else 0,\n"
    "            }"
)
if OLD_SNAP_POP in c:
    c = c.replace(OLD_SNAP_POP, NEW_SNAP_POP, 1); ok += 1
    print("snapshot metrics population added")
else:
    print("NOT FOUND: snapshot population block")

# ═══════════════════════════════════════════════════════════════════════════
# 3. REPLACE _export_dashboard_html() with JARVIS HUD version
# ═══════════════════════════════════════════════════════════════════════════
OLD_EXPORT_START = "def _export_dashboard_html():\n    \"\"\"Gera HTML standalone com todo o conteúdo do dashboard (Plotly + HTML).\"\"\""
OLD_EXPORT_END   = "    parts.append(\"</body></html>\")\n    return \"\\n\".join(parts)"

if OLD_EXPORT_START in c and OLD_EXPORT_END in c:
    start_idx = c.find(OLD_EXPORT_START)
    end_idx   = c.find(OLD_EXPORT_END) + len(OLD_EXPORT_END)

    NEW_EXPORT = r'''def _export_dashboard_html():
    """Exporta como JARVIS HUD HTML standalone — frame completo com dados reais."""
    if not _snapshot['sections']:
        return None

    import json as _json

    ticker  = _snapshot['ticker']
    spot    = _snapshot['spot']
    ts      = _snapshot['ts']
    m       = _snapshot.get('metrics', {})

    # ── Formatted metric strings ──────────────────────────────────────────
    _spot_s  = f"{spot:,.0f}"
    _flip_s  = f"{m.get('gamma_flip', 0):,.0f}" if m.get('gamma_flip') else "N/A"
    _gex_s   = f"{m.get('gex_net_bn', 0):+.1f}B"
    _pc_s    = f"{m.get('pc_ratio', 0):.2f}\u00d7"
    _ivrv_s  = f"{m.get('iv_rv_pp', 0):+.1f}pp"
    _sq_s    = f"{m.get('squeeze_score', 'N/A')}/100"
    _tail_s  = f"{m.get('tail_score', 0):.0f}/100"
    _iv_s    = f"{m.get('iv_30d', 0)*100:.2f}%"
    _rv_s    = f"{m.get('rv_30d', 0)*100:.2f}%"
    _cw_s    = f"{m.get('call_wall', 0):,.0f}" if m.get('call_wall') else "N/A"
    _pw_s    = f"{m.get('put_wall',  0):,.0f}" if m.get('put_wall')  else "N/A"

    # ── Build tab buttons + panel HTML ────────────────────────────────────
    _tab_btns   = []
    _tab_panels = []
    for i, sec in enumerate(_snapshot['sections']):
        _active = ' active' if i == 0 else ''
        _tab_btns.append(
            f"<button class='jv-tab{_active}' data-jvtab='jvtab{i}'>"
            f"{sec['name'].upper()}</button>")
        _items = []
        for item in sec['content']:
            if item['type'] == 'plotly':
                _fdiv = go.Figure(item['data']).to_html(
                    full_html=False, include_plotlyjs=False,
                    config={'displaylogo': False, 'responsive': True},
                    div_id=f"plt{i}_{len(_items)}")
                _items.append(
                    "<div class='jv-panel'>"
                    "<div class='ct tl'></div><div class='ct tr'></div>"
                    "<div class='cb bl'></div><div class='cb br'></div>"
                    f"{_fdiv}</div>")
            elif item['type'] == 'html':
                _items.append(
                    "<div class='jv-panel'>"
                    "<div class='ct tl'></div><div class='ct tr'></div>"
                    "<div class='cb bl'></div><div class='cb br'></div>"
                    f"<div class='mm-dash'>{item['data']}</div></div>")
            elif item['type'] == 'matplotlib':
                _items.append(
                    "<div class='jv-panel'>"
                    "<div class='ct tl'></div><div class='ct tr'></div>"
                    "<div class='cb bl'></div><div class='cb br'></div>"
                    f"<img src='data:image/png;base64,{item['data']}' "
                    f"style='max-width:100%;display:block;'></div>")
        _tab_panels.append(
            f"<div class='jv-tab-panel{_active}' id='jvtab{i}'>"
            + "\n".join(_items) + "</div>")

    _tabs_html   = "\n".join(_tab_btns)
    _panels_html = "\n".join(_tab_panels)

    # ── Ticker items ──────────────────────────────────────────────────────
    _tick_data = [
        (f"SPX \u25b2 {_spot_s}",   True),
        (f"GEX {_gex_s}",           False),
        (f"GAMMA FLIP {_flip_s}",   False),
        (f"IV 30D {_iv_s}",         False),
        (f"RV 30D {_rv_s}",         True),
        (f"P/C {_pc_s}",            False),
        (f"SQUEEZE {_sq_s}",        False),
        (f"TAIL RISK {_tail_s}",    False),
        (f"CALL WALL \u25b2 {_cw_s}", True),
        (f"PUT WALL \u25bc {_pw_s}", False),
        (f"IV\u2212RV {_ivrv_s}",   False),
    ]
    _tick_html = " &nbsp;\u00b7&nbsp; ".join(
        f"<span class='jt-item {'jt-up' if u else 'jt-dn'}'>{t}</span>"
        for t, u in _tick_data)

    # ════════════════════════════════════════════════════════════════════════
    # JARVIS EXPORT CSS (frame + layout + Stark panel classes)
    # ════════════════════════════════════════════════════════════════════════
    _jv_css = """
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');
    *, *::before, *::after { box-sizing:border-box; margin:0; padding:0; }
    body {
      background:#020810; color:rgba(0,212,232,.85);
      font-family:'Share Tech Mono',monospace;
      min-height:100vh; padding-bottom:32px;
      overflow-x:hidden;
    }
    @keyframes spin-cw  { from{transform:rotate(0deg)}   to{transform:rotate(360deg)} }
    @keyframes spin-ccw { from{transform:rotate(360deg)} to{transform:rotate(0deg)} }
    @keyframes jv-ticker { 0%{transform:translateX(0)} 100%{transform:translateX(-50%)} }
    @keyframes jv-pulse  { 0%,100%{opacity:1} 50%{opacity:.3} }
    @keyframes jv-scan   { 0%{left:-100%} 100%{left:200%} }
    @keyframes jv-boot-fade { to{opacity:0;pointer-events:none} }
    /* Particles & scanlines */
    #jv-pcv { position:fixed; inset:0; z-index:0; pointer-events:none; }
    #jv-sl  { position:fixed; inset:0; z-index:1; pointer-events:none;
      background:repeating-linear-gradient(to bottom,transparent 0,transparent 3px,
        rgba(0,0,0,.07) 3px,rgba(0,0,0,.07) 4px); }
    /* Boot overlay */
    #jv-boot {
      position:fixed; inset:0; background:#020810; z-index:9999;
      display:flex; flex-direction:column; align-items:center; justify-content:center;
      transition:opacity .8s;
    }
    .jb-line { font-family:'Share Tech Mono',monospace; font-size:12px;
      color:rgba(0,200,255,.75); opacity:0; margin:3px 0; transition:opacity .3s; }
    .jb-line.show { opacity:1; }
    /* App shell */
    #jv-app { position:relative; z-index:2; display:flex; flex-direction:column; min-height:100vh; }
    /* CMD strip */
    #jv-cmd {
      background:rgba(0,4,10,.97); border-bottom:1px solid rgba(0,212,232,.18);
      padding:4px 16px; display:flex; align-items:center; gap:0;
      font-size:10px; white-space:nowrap; overflow:hidden; flex-shrink:0;
    }
    .jv-cmd-item { padding:0 12px; border-right:1px solid rgba(0,212,232,.14); }
    .jv-cmd-item:first-child { padding-left:4px; }
    .jv-cmd-label { font-family:Orbitron,monospace; font-size:7px; letter-spacing:2px; opacity:.45; display:block; }
    .jv-cmd-val   { font-family:Orbitron,monospace; font-size:11px; font-weight:700; color:rgba(0,212,232,1); }
    .jv-live-dot  { width:6px; height:6px; background:rgba(0,212,232,1); border-radius:50%;
      display:inline-block; margin-right:6px; animation:jv-pulse 2s infinite;
      box-shadow:0 0 5px rgba(0,212,232,.8); }
    /* Header */
    #jv-header {
      background:rgba(0,4,12,.98); border-bottom:1px solid rgba(0,212,232,.2);
      padding:8px 16px; display:flex; align-items:center; gap:14px; position:relative;
      overflow:hidden; flex-shrink:0;
    }
    #jv-header::after {
      content:''; position:absolute; bottom:0; left:-100%; width:60%; height:1px;
      background:linear-gradient(90deg,transparent,rgba(0,212,232,.8),transparent);
      animation:jv-scan 4s linear infinite;
    }
    .jv-brand-title { font-family:Orbitron,monospace; font-size:20px; font-weight:900;
      letter-spacing:4px; color:rgba(0,212,232,1); text-shadow:0 0 18px rgba(0,212,232,.55); }
    .jv-brand-sub   { font-family:Orbitron,monospace; font-size:7px; letter-spacing:2px;
      opacity:.4; margin-top:2px; }
    /* Tabs */
    #jv-tabs { display:flex; gap:3px; margin-left:auto; flex-wrap:wrap; }
    .jv-tab {
      font-family:Orbitron,monospace; font-size:9px; letter-spacing:1.5px;
      padding:5px 12px; background:transparent; border:1px solid rgba(0,212,232,.2);
      color:rgba(0,212,232,.45); cursor:pointer;
      clip-path:polygon(0 0,calc(100% - 6px) 0,100% 6px,100% 100%,6px 100%,0 calc(100% - 6px));
      transition:all .2s;
    }
    .jv-tab.active, .jv-tab:hover {
      background:rgba(0,212,232,.1); border-color:rgba(0,212,232,.6); color:rgba(0,212,232,1);
    }
    #jv-clock { font-family:Orbitron,monospace; font-size:10px; font-weight:700;
      letter-spacing:2px; margin-left:12px; opacity:.75; white-space:nowrap; }
    /* Content */
    #jv-content { flex:1; padding:10px 14px; overflow-y:auto; }
    .jv-tab-panel { display:none; }
    .jv-tab-panel.active { display:block; }
    /* JARVIS panels */
    .jv-panel {
      background:rgba(0,6,18,.92); border:1px solid rgba(0,212,232,.18);
      clip-path:polygon(0 0,calc(100% - 10px) 0,100% 10px,100% 100%,10px 100%,0 calc(100% - 10px));
      padding:12px; margin-bottom:8px; position:relative;
    }
    .ct, .cb { position:absolute; width:10px; height:10px; border-color:rgba(0,212,232,.4); border-style:solid; }
    .ct.tl { top:-1px; left:-1px;  border-width:2px 0 0 2px; }
    .ct.tr { top:-1px; right:-1px; border-width:2px 2px 0 0; }
    .cb.bl { bottom:-1px; left:-1px;  border-width:0 0 2px 2px; }
    .cb.br { bottom:-1px; right:-1px; border-width:0 2px 2px 0; }
    /* Ticker */
    #jv-ticker {
      position:fixed; bottom:0; left:0; right:0; height:26px;
      background:rgba(0,4,10,.98); border-top:1px solid rgba(0,212,232,.18);
      overflow:hidden; display:flex; align-items:center; z-index:200;
    }
    #jv-ticker-track { display:flex; gap:24px; white-space:nowrap;
      animation:jv-ticker 40s linear infinite; }
    .jt-item { font-family:Orbitron,monospace; font-size:9px; letter-spacing:1px; }
    .jt-up   { color:rgba(0,212,232,1); }
    .jt-dn   { color:rgba(0,212,232,.38); }
    /* Arc reactor */
    .jv-r1 { animation:spin-cw  14s linear infinite; transform-origin:21px 21px; }
    .jv-r2 { animation:spin-ccw  9s linear infinite; transform-origin:21px 21px; }
    .jv-r3 { animation:spin-cw   5s linear infinite; transform-origin:21px 21px; }
    """

    # ════════════════════════════════════════════════════════════════════════
    # JARVIS EXPORT JS
    # ════════════════════════════════════════════════════════════════════════
    _jv_js = r"""
(function(){
  /* Particles */
  var pcv=document.getElementById('jv-pcv');
  if(!pcv) return;
  var ctx=pcv.getContext('2d'),W,H,pts=[];
  function resize(){W=pcv.width=window.innerWidth;H=pcv.height=window.innerHeight;}
  resize(); window.addEventListener('resize',resize);
  for(var i=0;i<50;i++) pts.push({x:Math.random()*W,y:Math.random()*H,
    vx:(Math.random()-.5)*.2,vy:(Math.random()-.5)*.2,r:Math.random()*1.2+.4});
  function draw(){
    ctx.clearRect(0,0,W,H);
    ctx.lineWidth=.5; ctx.strokeStyle='rgba(0,200,255,.04)';
    for(var x=0;x<W;x+=60){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
    for(var y=0;y<H;y+=60){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
    for(var i=0;i<pts.length;i++){
      for(var j=i+1;j<pts.length;j++){
        var dx=pts[i].x-pts[j].x,dy=pts[i].y-pts[j].y,d=Math.sqrt(dx*dx+dy*dy);
        if(d<110){ctx.strokeStyle='rgba(0,200,255,'+(0.07*(1-d/110))+')';
          ctx.beginPath();ctx.moveTo(pts[i].x,pts[i].y);ctx.lineTo(pts[j].x,pts[j].y);ctx.stroke();}
      }
      pts[i].x+=pts[i].vx;pts[i].y+=pts[i].vy;
      if(pts[i].x<0||pts[i].x>W)pts[i].vx*=-1;
      if(pts[i].y<0||pts[i].y>H)pts[i].vy*=-1;
      ctx.fillStyle='rgba(0,200,255,.45)';
      ctx.beginPath();ctx.arc(pts[i].x,pts[i].y,pts[i].r,0,Math.PI*2);ctx.fill();
    }
    requestAnimationFrame(draw);
  }
  draw();
  /* Boot */
  var boot=document.getElementById('jv-boot'),log=document.getElementById('jv-boot-log');
  var lines=['Inicializando nucleo de risco...','Carregando superficie de vol...',
    'Conectando feed OI...','Compilando GEX matrix...','Calibrando modelos de cauda...',
    'Sincronizando posicionamento CTA...','Sistema operacional \u2014 ONLINE'];
  if(boot&&log){
    lines.forEach(function(t,i){
      setTimeout(function(){
        var d=document.createElement('div');d.className='jb-line';
        d.textContent='> '+t;log.appendChild(d);
        setTimeout(function(){d.classList.add('show');},20);
        if(i===lines.length-1)setTimeout(function(){
          boot.style.opacity='0';boot.style.pointerEvents='none';
          setTimeout(function(){boot.style.display='none';},820);
        },350);
      },i*240);
    });
  }
  /* Tabs */
  document.querySelectorAll('.jv-tab').forEach(function(t){
    t.addEventListener('click',function(){
      document.querySelectorAll('.jv-tab').forEach(function(x){x.classList.remove('active');});
      document.querySelectorAll('.jv-tab-panel').forEach(function(x){x.classList.remove('active');});
      t.classList.add('active');
      var el=document.getElementById(t.getAttribute('data-jvtab'));
      if(el) el.classList.add('active');
    });
  });
  /* Clock */
  function tick(){
    var el=document.getElementById('jv-clock');
    if(el) el.textContent=new Date().toLocaleTimeString('pt-BR',{hour12:false});
  }
  setInterval(tick,1000); tick();
  /* Ticker */
  var tt=document.getElementById('jv-ticker-track');
  if(tt){var h=tt.innerHTML;tt.innerHTML=h+' &nbsp;\u00b7&nbsp; '+h;}
})();
"""

    # ════════════════════════════════════════════════════════════════════════
    # ASSEMBLE FULL HTML
    # ════════════════════════════════════════════════════════════════════════
    html = (
        "<!DOCTYPE html><html lang='pt-BR'>\n"
        "<head><meta charset='UTF-8'/>\n"
        f"<title>JARVIS \u2014 {ticker} @ {spot:,.2f} \u00b7 {ts}</title>\n"
        "<script src='https://cdn.plot.ly/plotly-2.27.0.min.js'></script>\n"
        f"<style>{_jv_css}</style>\n"
        f"{DASH_CSS.replace('<style>', '<style>.mm-dash{{position:relative}}')}\n"
        "</head>\n<body>\n"

        # Canvas + scanlines + boot
        "<canvas id='jv-pcv'></canvas>\n"
        "<div id='jv-sl'></div>\n"
        "<div id='jv-boot'>\n"
        "  <svg width='130' height='130' viewBox='0 0 130 130'>\n"
        "    <defs><filter id='jbg'><feGaussianBlur stdDeviation='4' result='b'/>"
        "<feMerge><feMergeNode in='b'/><feMergeNode in='SourceGraphic'/></feMerge></filter></defs>\n"
        "    <circle cx='65' cy='65' r='52' fill='none' stroke='rgba(0,200,255,.28)' stroke-width='1' stroke-dasharray='8 5' class='jv-r1'/>\n"
        "    <circle cx='65' cy='65' r='38' fill='none' stroke='rgba(0,200,255,.45)' stroke-width='1' stroke-dasharray='5 4' class='jv-r2'/>\n"
        "    <circle cx='65' cy='65' r='24' fill='none' stroke='rgba(0,200,255,.65)' stroke-width='1' stroke-dasharray='3 3' class='jv-r3'/>\n"
        "    <circle cx='65' cy='65' r='8' fill='rgba(0,200,255,.95)' filter='url(#jbg)'/>\n"
        "    <circle cx='65' cy='65' r='3' fill='white' opacity='.8'/>\n"
        "  </svg>\n"
        "  <div style='font-family:Orbitron,monospace;font-size:16px;font-weight:900;"
        "letter-spacing:5px;color:rgba(0,200,255,1);margin-top:12px;"
        "text-shadow:0 0 16px rgba(0,200,255,.6)'>J.A.R.V.I.S</div>\n"
        "  <div style='font-family:Orbitron,monospace;font-size:7px;letter-spacing:2px;"
        f"color:rgba(0,200,255,.4);margin-top:3px'>{ticker} \u00b7 OPTIONS ANALYTICS \u00b7 {ts}</div>\n"
        "  <div id='jv-boot-log' style='margin-top:16px;text-align:left;width:340px;min-height:110px'></div>\n"
        "</div>\n"

        # App shell
        "<div id='jv-app'>\n"

        # CMD strip
        "<div id='jv-cmd'>\n"
        "  <div class='jv-cmd-item' style='padding-left:0;display:flex;align-items:center'>\n"
        "    <span class='jv-live-dot'></span>\n"
        "    <span class='jv-cmd-label' style='display:inline'>SPX MARKET COMMAND</span></div>\n"
        f"  <div class='jv-cmd-item'><span class='jv-cmd-label'>SPOT</span><span class='jv-cmd-val'>{_spot_s}</span></div>\n"
        f"  <div class='jv-cmd-item'><span class='jv-cmd-label'>GAMMA FLIP</span><span class='jv-cmd-val'>{_flip_s}</span></div>\n"
        f"  <div class='jv-cmd-item'><span class='jv-cmd-label'>GEX NET</span><span class='jv-cmd-val'>{_gex_s}</span></div>\n"
        f"  <div class='jv-cmd-item'><span class='jv-cmd-label'>P/C RATIO</span><span class='jv-cmd-val'>{_pc_s}</span></div>\n"
        f"  <div class='jv-cmd-item'><span class='jv-cmd-label'>IV\u2212RV</span><span class='jv-cmd-val'>{_ivrv_s}</span></div>\n"
        f"  <div class='jv-cmd-item'><span class='jv-cmd-label'>SQUEEZE</span><span class='jv-cmd-val'>{_sq_s}</span></div>\n"
        f"  <div class='jv-cmd-item'><span class='jv-cmd-label'>TAIL RISK</span><span class='jv-cmd-val'>{_tail_s}</span></div>\n"
        "</div>\n"

        # Header
        "<div id='jv-header'>\n"
        "  <div style='flex-shrink:0'>\n"
        "    <svg width='44' height='44' viewBox='0 0 44 44'>\n"
        "      <defs><filter id='rfg'><feGaussianBlur stdDeviation='2' result='b'/>"
        "<feMerge><feMergeNode in='b'/><feMergeNode in='SourceGraphic'/></feMerge></filter></defs>\n"
        "      <circle cx='22' cy='22' r='19' fill='none' stroke='rgba(0,200,255,.22)' stroke-width='1' stroke-dasharray='6 4' class='jv-r1'/>\n"
        "      <circle cx='22' cy='22' r='14' fill='none' stroke='rgba(0,200,255,.42)' stroke-width='1' stroke-dasharray='4 3' class='jv-r2'/>\n"
        "      <circle cx='22' cy='22' r='8'  fill='none' stroke='rgba(0,200,255,.62)' stroke-width='1' stroke-dasharray='2 2' class='jv-r3'/>\n"
        "      <circle cx='22' cy='22' r='3.5' fill='rgba(0,200,255,.95)' filter='url(#rfg)'/>\n"
        "    </svg>\n"
        "  </div>\n"
        "  <div>\n"
        "    <div class='jv-brand-title'>J.A.R.V.I.S</div>\n"
        "    <div class='jv-brand-sub'>JUST A RATHER VERY INTELLIGENT SYSTEM \u00b7 OPTIONS CORE \u00b7 v4.2</div>\n"
        "  </div>\n"
        f"  <div id='jv-tabs'>{_tabs_html}</div>\n"
        "  <div id='jv-clock'></div>\n"
        "</div>\n"

        # Content
        f"<div id='jv-content'>{_panels_html}</div>\n"

        "</div>\n"  # /jv-app

        # Ticker
        "<div id='jv-ticker'>\n"
        f"  <div id='jv-ticker-track'>{_tick_html}</div>\n"
        "</div>\n"

        f"<script>{_jv_js}</script>\n"
        "</body></html>"
    )

    return html'''

    c = c[:start_idx] + NEW_EXPORT + c[end_idx:]
    ok += 1
    print("_export_dashboard_html() replaced with JARVIS version")
else:
    print("NOT FOUND: export function boundaries")
    if OLD_EXPORT_START not in c:
        print("  - missing: function start")
    if OLD_EXPORT_END not in c:
        print("  - missing: function end")

print(f"\nApplied {ok}/3 changes")
with open(r'C:/Users/rafael bentes/bbg/examples/greeks_dashboard.py', 'w', encoding='utf-8') as f:
    f.write(c)
print("Done.")
