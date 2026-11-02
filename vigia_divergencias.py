#!/usr/bin/env python3
"""Vigia de divergencias MACD + Fibonacci (operativa atribuida a David Trullas) -> avisos por Telegram.

Corre en la nube (GitHub Actions, revisando cada minuto) para avisar aunque el Mac este cerrado.
Mismas reglas que backtest/trullas_divergencias.py y tradingview/Divergencias_Fibo_Trullas.pine, en velas de 1, 5 y 15 min y 1 h:
  - Dos pivotes consecutivos (3 velas a cada lado): precio hace nuevo extremo y la linea MACD no.
  - Orden stop 1 tick tras el 23,6 % del tramo local; SL 1 tick tras el extremo; T1 38,2 %; T2 61,8 %.
  - Confirmaciones: MACD (siempre) + divergencia RSI(14) + menos volumen (en divisas no hay volumen).
  - La orden se cancela si se hace nuevo extremo antes de entrar o tras 60 velas.
  - Solo se avisa con el filtro completo: 3/3 (MACD + RSI + volumen); 2/2 en divisas (sin volumen). En 1 min, sin avisos previos.
Avisos: 🆕 orden preparada · ✅ activada · 🎯 T1 / T2 · ⛔ stop · ❌ cancelada / caducada · 📋 informe del dia.
Cada aviso lleva el resultado de la auditoria de ese activo (2023-2026, con costes).

Datos: Yahoo Finance (1m, 5m, 15m, 60m). Oro: futuro GC=F desplazado al precio contado de Swissquote (XAUUSD de MT5).
Uso: python3 vigia_divergencias.py [--prueba] [--informe] [--bucle MINUTOS]   (--prueba: muestra los avisos sin enviar ni guardar)
Telegram: variables TELEGRAM_TOKEN y TELEGRAM_CHAT, o config_telegram.json junto al script.
"""
import json, math, os, sys, time, urllib.request, urllib.parse, datetime as dt
import numpy as np, pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
ESTADO = os.environ.get('VIGIA_ESTADO') or os.path.join(AQUI, 'data', 'vigia_divergencias.json')
UA = {'User-Agent': 'Mozilla/5.0'}
SQ = 'https://forex-data-feed.swissquote.com/public-quotes/bboquotes/instrument/XAU/USD'
CAPITAL, RIESGO = 820.0, 0.01
FONDEO, FONDEO_CAP, FONDEO_RIESGO = 'FTMO 10K', 10000.0, 0.015   # cuenta de fondeo: lotes y $ en cada aviso
N, ESPERA, MIN_SEP, MAX_SEP = 3, 60, 5, 120
F_ENT, F_T1, F_T2 = 0.236, 0.382, 0.618
VENTANA = pd.Timedelta(hours=3)        # no se avisa de eventos mas antiguos (primer arranque, retrasos)
# temporalidad: (minutos, intervalo y rango de Yahoo, nombre, avisos previos, auditoria 2023-2026 de esa temporalidad)
TFS = {
    '1m':  (1, '1m', '2d', '1 min', False,
            '⚠️ 1 MIN: NO RENTABLE en el backtest (oro PF 0,03-0,73; Nasdaq 0,23-0,86): el coste se come la ventaja. Solo demo.'),
    '5m':  (5, '5m', '5d', '5 min', True, None),       # usa la auditoria de cada activo
    '15m': (15, '15m', '1mo', '15 min', True,
            '⚠️ 15 MIN: NO VALIDADA (oro PF 0,83-1,22; Nasdaq 0,69-1,16; divisas 1,03-1,15): resultado cercano a cero. Solo demo.'),
    '1h':  (60, '60m', '3mo', '1 h', True,
            '⚠️ 1 HORA: NO VALIDADA (oro PF 0,65-1,48; Nasdaq 0,54-0,72; divisas 0,46-1,12) y muy pocas señales. Solo demo.'),
}
SEP = '━━━━━━━━━━━━━━━━━━'

# simbolo Yahoo, nombre, tick, decimales, confirmaciones minimas, tipo de tamano, veredicto de la auditoria
ACTIVOS = {
    'QQQ':      dict(yahoo='QQQ', nombre='Nasdaq 100', tick=0.01, dec=2, minconf=3, tipo='accion', coste=0.03, apal=5,
                     audit='🧪 EN OBSERVACIÓN · Nasdaq M5 3/3 en sesión EE. UU.: PF 1,33 / 1,52 (52 ops 2023-26). Prometedora, no validada.'),
    'XAUUSD':   dict(yahoo='GC=F', nombre='Oro contado', tick=0.01, dec=2, minconf=3, tipo='oro', coste=0.40, apal=20,
                     audit='⚠️ NO RENTABLE en el backtest del oro (M5 3/3: PF 0,59 / 1,41). Solo para seguir en demo.'),
    'EURUSD':   dict(yahoo='EURUSD=X', nombre='EUR/USD', tick=0.00001, dec=5, minconf=3, tipo='fx', coste=0.00012, apal=30,
                     audit='⚠️ NO RENTABLE en el backtest (M5: PF 0,45-0,89). Solo para seguir en demo.'),
    'GBPUSD':   dict(yahoo='GBPUSD=X', nombre='GBP/USD', tick=0.00001, dec=5, minconf=3, tipo='fx', coste=0.00015, apal=30,
                     audit='⚠️ NO VALIDADA en el backtest (M5: PF 0,47-1,26). Solo para seguir en demo.'),
    'USDJPY':   dict(yahoo='USDJPY=X', nombre='USD/JPY', tick=0.001, dec=3, minconf=3, tipo='fx', coste=0.015, apal=30,
                     audit='⚠️ NO RENTABLE en el backtest (M5: PF 0,70-1,01). Solo para seguir en demo.'),
}


# ---------------------------------------------------------------- datos
def _get(url, intentos=3):
    for i in range(intentos):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20).read()
        except Exception:
            if i == intentos - 1:
                raise
            time.sleep(3 * (i + 1))


def velas(yahoo, tf='5m'):
    mins, intervalo, rango = TFS[tf][:3]
    r = json.loads(_get(f'https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(yahoo)}'
                        f'?interval={intervalo}&range={rango}'))['chart']['result'][0]
    q = r['indicators']['quote'][0]
    d = pd.DataFrame({'Open': q['open'], 'High': q['high'], 'Low': q['low'], 'Close': q['close'],
                      'Vol': q['volume']}, index=pd.to_datetime(r['timestamp'], unit='s', utc=True))
    d.index = d.index.floor(f'{mins}min')
    d = d[~d.index.duplicated(keep='last')].dropna(subset=['Open', 'High', 'Low', 'Close'])
    d['Vol'] = d.Vol.fillna(0)
    ahora = pd.Timestamp.now(tz='UTC')
    return d[d.index + pd.Timedelta(minutes=mins) <= ahora]       # solo velas cerradas


def velas_5m(yahoo):
    return velas(yahoo, '5m')


def oro_contado():
    datos = json.loads(_get(SQ))
    p = datos[0]['spreadProfilePrices'][0]
    return (p['bid'] + p['ask']) / 2


# ---------------------------------------------------------------- reglas
def indicadores(d):
    e = lambda s, n: s.ewm(span=n, adjust=False).mean()
    d = d.copy(); d['macd'] = e(d.Close, 12) - e(d.Close, 26)
    dl = d.Close.diff(); up = dl.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    dn = (-dl.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    d['rsi'] = 100 - 100 / (1 + up / dn)
    return d


def pivotes(x, alto):
    v = np.lib.stride_tricks.sliding_window_view(x, 2 * N + 1)
    c = v[:, N]
    ok = (c == v.max(1)) if alto else (c == v.min(1))
    out = np.zeros(len(x), bool); out[N:len(x) - N] = ok
    return out


def setups(d, a, mins=5):
    """Configuraciones con su historia de eventos [(tiempo, evento, precio)]; solo informacion pasada en cada vela."""
    H, L, C, Mc, Rs, V = d.High.values, d.Low.values, d.Close.values, d.macd.values, d.rsi.values, d.Vol.values
    idx, n, tick, hay_vol = d.index, len(d), a['tick'], (d.Vol > 0).mean() > 0.5
    out = []
    for lado in (-1, 1):
        piv = np.nonzero(pivotes(H if lado < 0 else L, alto=lado < 0))[0]
        for p, b in zip(piv[:-1], piv[1:]):
            if not (MIN_SEP <= b - p <= MAX_SEP):
                continue
            p1, p2 = (H[p], H[b]) if lado < 0 else (L[p], L[b])
            if not ((p2 > p1) if lado < 0 else (p2 < p1)) or not ((Mc[b] < Mc[p]) if lado < 0 else (Mc[b] > Mc[p])):
                continue
            c_rsi = (Rs[b] < Rs[p]) if lado < 0 else (Rs[b] > Rs[p])
            c_vol = hay_vol and V[b] < V[p]
            conf, maxconf = 1 + c_rsi + c_vol, 3 if hay_vol else 2
            if conf < min(a['minconf'], maxconf):
                continue
            medio = L[p:b + 1].min() if lado < 0 else H[p:b + 1].max()
            tramo = abs(p2 - medio)
            ini = b + N + 1
            if tramo <= 0 or ini >= n + 1:
                continue
            s = dict(lado=lado, ext=p2, conf=int(conf), maxconf=maxconf, R=0.0,
                     ent=p2 + lado * (F_ENT * tramo + tick), sl=p2 - lado * tick,
                     t1=p2 + lado * F_T1 * tramo, t2=p2 + lado * F_T2 * tramo,
                     id=f"{'V' if lado < 0 else 'C'}{idx[b]:%Y%m%d%H%M}")
            s['R'] = abs(s['ent'] - s['sl'])
            t0 = idx[ini - 1] + pd.Timedelta(minutes=mins)      # cierre de la vela que confirma el pivote
            if (C[ini - 1] <= s['ent']) if lado < 0 else (C[ini - 1] >= s['ent']):
                continue                                       # ya habia pasado el nivel: no se persigue
            ev = [(t0, 'nueva', None)]
            j, estado = ini, 'pendiente'
            while j < n and estado not in ('stop', 't2', 'cancelada', 'caducada'):
                hi, lo, tj = H[j], L[j], idx[j] + pd.Timedelta(minutes=mins)
                if estado == 'pendiente':
                    if (hi > p2) if lado < 0 else (lo < p2):
                        estado = 'cancelada'; ev.append((tj, estado, None))
                    elif (lo <= s['ent']) if lado < 0 else (hi >= s['ent']):
                        estado = 'activada'; ev.append((tj, estado, s['ent']))
                    elif j - ini >= ESPERA:
                        estado = 'caducada'; ev.append((tj, estado, None))
                else:
                    if (hi >= s['sl']) if lado < 0 else (lo <= s['sl']):
                        estado = 'stop'; ev.append((tj, estado, s['sl']))
                    elif (lo <= s['t2']) if lado < 0 else (hi >= s['t2']):
                        if estado != 't1':
                            ev.append((tj, 't1', s['t1']))
                        estado = 't2'; ev.append((tj, estado, s['t2']))
                    elif estado == 'activada' and ((lo <= s['t1']) if lado < 0 else (hi >= s['t1'])):
                        estado = 't1'; ev.append((tj, estado, s['t1']))
                j += 1
            s['ev'], s['estado'] = ev, estado
            out.append(s)
    return out


def vigilancia(d, a, mins=5):
    """Aviso previo (sin orden todavia): divergencia formandose que ya cumple el filtro completo (MACD + RSI + volumen)."""
    H, L, C, Mc, idx, n = d.High.values, d.Low.values, d.Close.values, d.macd.values, d.index, len(d)
    Rs, V, hay_vol = d.rsi.values, d.Vol.values, (d.Vol > 0).mean() > 0.5
    tr = np.maximum(H[1:] - L[1:], np.maximum(abs(H[1:] - C[:-1]), abs(L[1:] - C[:-1])))
    atr = tr[-14:].mean(); out = []
    for lado in (-1, 1):
        X = H if lado < 0 else L
        piv = np.nonzero(pivotes(X, alto=lado < 0))[0]
        if not len(piv):
            continue
        p = piv[-1]; nom = 'VENTA' if lado < 0 else 'COMPRA'; ext = 'máximo' if lado < 0 else 'mínimo'
        mas_alla = (lambda x, y: x > y) if lado < 0 else (lambda x, y: x < y)
        debil = (lambda m: m < Mc[p]) if lado < 0 else (lambda m: m > Mc[p])
        completo = lambda k: ((Rs[k] < Rs[p]) if lado < 0 else (Rs[k] > Rs[p])) and (not hay_vol or V[k] < V[p])
        forma = None
        for k in range(n - 1, n - N - 1, -1):            # vela candidata a pivote, sin las N velas de la derecha
            if MIN_SEP <= k - p <= MAX_SEP and X[k] == (X[k - N:].max() if lado < 0 else X[k - N:].min()) \
                    and mas_alla(X[k], X[p]) and debil(Mc[k]) and completo(k):
                forma = k; break
        clave = f"{'V' if lado < 0 else 'C'}{idx[p]:%Y%m%d%H%M}"
        if forma is not None:
            k = forma; medio = L[p:k + 1].min() if lado < 0 else H[p:k + 1].max(); tramo = abs(X[k] - medio)
            faltan = N - (n - 1 - k)
            out.append((f'forma|{clave}', '\n'.join([
                f"👀 <b>POSIBLE {nom} FORMÁNDOSE</b> · 5 min",
                f"Nuevo {ext} <code>{fmt(a, X[k])}</code> más allá del anterior <code>{fmt(a, X[p])}</code>, con el MACD más débil (divergencia).",
                f"Si en las próximas {faltan} velas (≈{mins * faltan} min) no supera <code>{fmt(a, X[k])}</code>, saldrá la orden:",
                f"   entrada ≈ <code>{fmt(a, X[k] + lado * 0.236 * tramo)}</code> · SL ≈ <code>{fmt(a, X[k] - lado * a['tick'])}</code> · "
                f"T2 ≈ <code>{fmt(a, X[k] + lado * 0.618 * tramo)}</code>",
                "Cumple ya el filtro completo (MACD, RSI y volumen). Estate atento; aún no hay orden."])))
    return out


# ---------------------------------------------------------------- mensajes
def tamano(a, R, precio):
    """Unidades para arriesgar el 1 %, limitadas por el apalancamiento minorista (ESMA) de cada activo."""
    obj = CAPITAL * RIESGO
    if a['tipo'] == 'accion':
        k = min(max(1, int(obj // R)), int(CAPITAL * a['apal'] // precio))
        return f"{k} acciones → riesgo ≈ {k * R:.0f} $ ({k * R / CAPITAL:.1%})"
    if a['tipo'] == 'oro':                                    # 0,01 lotes = 1 oz
        usd_ud, ud_max = R, CAPITAL * a['apal'] / precio
    else:                                                     # 0,01 lotes = 1.000 unidades
        usd_ud = R * 1000 / (precio if a['yahoo'].startswith('USD') else 1)
        ud_max = CAPITAL * a['apal'] / (1000 * (1 if a['yahoo'].startswith('USD') else precio))
    k = max(1, min(int(obj // usd_ud), int(ud_max)))
    tope = ' (tope por apalancamiento)' if int(obj // usd_ud) > int(ud_max) else ''
    return f"{k / 100:.2f} lotes → riesgo ≈ {k * usd_ud:.0f} $ ({k * usd_ud / CAPITAL:.1%}){tope}"


def fondeo(a, s):
    """Lotes y dolares en la cuenta de fondeo (CFD: 1 lote oro = 100 oz, 1 lote divisas = 100.000 unidades)."""
    if a['tipo'] == 'accion':
        return "🏦 En la cuenta de fondeo se opera el US100 (otros precios): usa el % de riesgo, no estos niveles."
    if a['tipo'] == 'oro':
        usd_lote, nocional = s['R'] * 100, 100 * s['ent']
    else:
        usd_lote = s['R'] * 100000 / (s['ent'] if a['yahoo'].startswith('USD') else 1)
        nocional = 100000 * (1 if a['yahoo'].startswith('USD') else s['ent'])
    lotes = math.floor(FONDEO_CAP * FONDEO_RIESGO / usd_lote * 100) / 100
    tope = math.floor(FONDEO_CAP * a['apal'] / nocional * 100) / 100     # mismo apalancamiento prudente que la cuenta propia
    nota = ' (tope de apalancamiento)' if lotes > tope else ''
    lotes = max(0.01, min(lotes, tope))
    sl = lotes * usd_lote
    g = lambda x: sl * abs(x - s['ent']) / s['R']
    return (f"🏦 <b>{FONDEO}</b> ({FONDEO_RIESGO:.1%}): <b>{lotes:.2f} lotes</b>{nota} · SL <b>−{sl:.0f} $</b> · "
            f"T1 <b>+{g(s['t1']):.0f} $</b> · T2 <b>+{g(s['t2']):.0f} $</b>")


def fmt(a, x):
    return f"{x:.{a['dec']}f}"


def mensaje(sim, a, s, evento, t, precio, tfl='5 min', audit=None):
    lado = 'VENTA' if s['lado'] < 0 else 'COMPRA'
    hora = t.tz_convert('Europe/Madrid').strftime('%H:%M')
    rr = lambda x: abs(x - s['ent']) / s['R'] if s['R'] else 0
    if evento == 'nueva':
        return '\n'.join([
            f"🆕 <b>{'🔴' if s['lado'] < 0 else '🟢'} {lado} PREPARADA · {sim}</b> ({a['nombre']}) · <b>{tfl}</b> · {hora}",
            f"Divergencia {'bajista' if s['lado'] < 0 else 'alcista'} MACD · confirmaciones {s['conf']}/{s['maxconf']} "
            + ('⭐ filtro completo' if s['conf'] == s['maxconf'] else '(filtro incompleto: menos fiable)'),
            SEP,
            f"👉 Orden {'sell stop' if s['lado'] < 0 else 'buy stop'} en <code>{fmt(a, s['ent'])}</code> (23,6 %)",
            f"⛔ SL: <code>{fmt(a, s['sl'])}</code> (riesgo {fmt(a, s['R'])})",
            f"🎯 T1 38,2 %: <code>{fmt(a, s['t1'])}</code> ({rr(s['t1']):.1f}R)",
            f"🎯 T2 61,8 %: <code>{fmt(a, s['t2'])}</code> ({rr(s['t2']):.1f}R)",
            f"📦 Tamaño (1 % de {CAPITAL:.0f} $): {tamano(a, s['R'], s['ent'])}",
            fondeo(a, s),
            *([f"⚠️ Stop muy corto: el spread (~{fmt(a, a['coste'])}) es el {a['coste'] / s['R']:.0%} del riesgo."]
              if a['coste'] > 0.25 * s['R'] else []),
            f"⏳ Cancelar si hace nuevo {'máximo' if s['lado'] < 0 else 'mínimo'} ({fmt(a, s['ext'])}) o tras 60 velas sin entrar.",
            SEP, audit or a['audit']])
    textos = {
        'activada': f"✅ <b>{lado} ACTIVADA · {sim}</b> en <code>{fmt(a, s['ent'])}</code> ({hora}). SL <code>{fmt(a, s['sl'])}</code> · T1 <code>{fmt(a, s['t1'])}</code> · T2 <code>{fmt(a, s['t2'])}</code>",
        't1': f"🎯 <b>T1 · {sim}</b> {lado.lower()}: tocado <code>{fmt(a, s['t1'])}</code> (+{rr(s['t1']):.1f}R) a las {hora}.",
        't2': f"🎯🎯 <b>T2 · {sim}</b> {lado.lower()}: tocado <code>{fmt(a, s['t2'])}</code> (+{rr(s['t2']):.1f}R) a las {hora}. Operación terminada.",
        'stop': f"⛔ <b>STOP · {sim}</b> {lado.lower()}: tocado <code>{fmt(a, s['sl'])}</code> (−1R) a las {hora}.",
        'cancelada': f"❌ <b>CANCELADA · {sim}</b> {lado.lower()} {fmt(a, s['ent'])}: nuevo extremo antes de entrar. Quita la orden.",
        'caducada': f"⌛ <b>CADUCADA · {sim}</b> {lado.lower()} {fmt(a, s['ent'])}: 60 velas sin activarse. Quita la orden.",
    }
    return textos[evento].replace(f"· {sim}</b>", f"· {sim} · {tfl}</b>", 1)


# ---------------------------------------------------------------- telegram y estado
def tg(texto):
    tok, chat = os.environ.get('TELEGRAM_TOKEN'), os.environ.get('TELEGRAM_CHAT')
    if not tok:
        c = json.load(open(os.path.join(AQUI, 'config_telegram.json')))
        tok, chat = c['token'], c['chat_id']
    datos = urllib.parse.urlencode({'chat_id': chat, 'text': texto, 'parse_mode': 'HTML'}).encode()
    urllib.request.urlopen(f'https://api.telegram.org/bot{tok}/sendMessage', data=datos, timeout=30).read()
    if 'XAUUSD' in texto:
        copia_github(texto)


def copia_github(texto):
    """En la nube: copia los avisos del oro como comentario de la issue AVISOS_ISSUE (la lee Claude)."""
    n, repo, token = (os.environ.get(k) for k in ('AVISOS_ISSUE', 'GITHUB_REPOSITORY', 'GITHUB_TOKEN'))
    if not (n and repo and token):
        return
    try:
        import re, html
        plano = html.unescape(re.sub('<[^>]+>', '', texto))
        cuerpo = json.dumps({'body': f'`{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC`\n\n{plano}'}).encode()
        req = urllib.request.Request(f'https://api.github.com/repos/{repo}/issues/{n}/comments', data=cuerpo,
                                     headers={'Authorization': f'Bearer {token}',
                                              'Accept': 'application/vnd.github+json'})
        urllib.request.urlopen(req, timeout=15).read()
    except Exception as e:
        print('Error copiando el aviso a GitHub:', e, flush=True)


def cargar_estado():
    try:
        return json.load(open(ESTADO))
    except Exception:
        return {'enviados': [], 'informe': ''}


def main(prueba=False, informe=False, todas=True):
    estado = cargar_estado(); enviados = set(estado['enviados'])
    ahora = pd.Timestamp.now(tz='UTC'); salida, hoy, errores = [], [], []
    madrid = ahora.tz_convert('Europe/Madrid')
    toca_informe = informe or (madrid.hour >= 22 and estado.get('informe') != str(madrid.date()) and madrid.weekday() < 5)
    minuto = int(ahora.timestamp() // 60)
    base_oro = None
    for tf, (mins, _, _, tfl, previos, audit_tf) in TFS.items():
        # cada temporalidad se descarga justo despues de cerrar su vela (y siempre en la primera vuelta y el informe)
        if not (todas or toca_informe or minuto % mins == 0):
            continue
        for sim, a in ACTIVOS.items():
            try:
                d = velas(a['yahoo'], tf)
                if sim == 'XAUUSD':                           # futuro GC -> precio contado (lo que ves en MT5)
                    try:
                        if base_oro is None:
                            base_oro = oro_contado() - d.Close.iloc[-1]
                        d[['Open', 'High', 'Low', 'Close']] += base_oro
                    except Exception:
                        errores.append('oro contado no disponible: niveles del futuro GC')
                di = indicadores(d)
                audit = audit_tf or a['audit']
                pre = '' if tf == '5m' else f'{tf}|'          # 5 min conserva las claves antiguas (no repetir avisos)
                aa = a                                          # todas: solo filtro completo (minconf=3)
                if previos:
                    for clave, texto in vigilancia(di, a, mins):
                        clave = f'{sim}|{pre}{clave}'
                        if clave not in enviados:
                            salida.append((ahora, texto.replace('</b> · 5 min', f"</b> · {sim} ({a['nombre']}) · {tfl}", 1) + '\n' + audit))
                            enviados.add(clave)
                for s in setups(di, aa, mins):
                    for t, evento, precio in s['ev']:
                        clave = f"{sim}|{pre}{s['id']}|{evento}"
                        if t.date() == ahora.date():
                            hoy.append((sim, s, evento, tfl))
                        if clave in enviados or ahora - t > VENTANA:
                            continue
                        salida.append((t, mensaje(sim, a, s, evento, t, precio, tfl, audit)))
                        enviados.add(clave)
            except Exception as e:
                errores.append(f'{sim} {tf}: {e}')
    salida.sort(key=lambda x: x[0])
    mensajes = [m for _, m in salida]
    if toca_informe:
        mensajes.append(texto_informe(hoy, errores))
        estado['informe'] = str(madrid.date())
    if prueba:
        print('\n\n'.join(mensajes) or 'Sin avisos nuevos.')
        print('Errores:', errores) if errores else None
        return
    for m in mensajes:
        tg(m)
        print('ENVIADO', ahora.strftime('%Y-%m-%d %H:%M'), m.split(chr(10))[0][:120], flush=True)
    estado['enviados'] = sorted(enviados)[-3000:]
    os.makedirs(os.path.dirname(ESTADO), exist_ok=True)
    json.dump(estado, open(ESTADO, 'w'))
    if errores:
        print('Errores:', errores)


def texto_informe(hoy, errores):
    lineas = [f"📋 <b>INFORME DIVERGENCIAS · {dt.date.today():%d/%m}</b>", SEP]
    finales = {}
    for sim, s, evento, tfl in hoy:
        finales[(sim, tfl, s['id'])] = (sim, s, evento, tfl)
    if not finales:
        lineas.append('Hoy no hubo divergencias que cumplan las reglas.')
    total = 0.0
    for sim, s, evento, tfl in sorted(finales.values(), key=lambda x: (x[3], x[0])):
        r = {'t2': abs(s['t2'] - s['ent']) / s['R'], 'stop': -1.0}.get(evento)
        if r is not None:
            total += r
        nom = {'nueva': 'pendiente', 'activada': 'abierta', 't1': 'abierta (T1 tocado)', 't2': 'T2 ✅',
               'stop': 'stop ⛔', 'cancelada': 'cancelada', 'caducada': 'caducada'}[evento]
        lineas.append(f"• {sim} {tfl} {'venta' if s['lado'] < 0 else 'compra'} {s['conf']}/{s['maxconf']}: {nom}")
    if finales:
        lineas.append(f"\nResultado de las cerradas hoy (objetivo T2): {total:+.1f}R")
    lineas.append('\nRecuerda: solo QQQ (5 min, 3/3, sesión EE. UU.) pasó la auditoría, y con muestra pequeña; las señales con menos confirmaciones son menos fiables.')
    if errores:
        lineas.append('⚠️ ' + '; '.join(errores))
    return '\n'.join(lineas)


def mercado_abierto(t=None):
    """Oro y divisas: de domingo 22:00 a viernes 22:00 UTC."""
    t = t or dt.datetime.now(dt.timezone.utc)
    d, h = t.weekday(), t.hour            # lunes = 0 ... domingo = 6
    return not (d == 5 or (d == 6 and h < 22) or (d == 4 and h >= 22))


def bucle(minutos):
    """Revisa cada minuto, 3 s despues del cierre; cada temporalidad se descarga al cerrar su vela (a la vez que TradingView)."""
    fin = time.time() + minutos * 60; primera = True
    while time.time() < fin:
        try:
            if mercado_abierto():
                main(todas=primera)
                primera = False
        except Exception as e:
            print('Error:', e)
        ahora = time.time()
        siguiente = (ahora // 60 + 1) * 60 + 3          # 3 s despues del proximo minuto
        time.sleep(max(1, siguiente - time.time()))


if __name__ == '__main__':
    if '--bucle' in sys.argv:
        bucle(float(sys.argv[sys.argv.index('--bucle') + 1]))
    else:
        main(prueba='--prueba' in sys.argv, informe='--informe' in sys.argv)
