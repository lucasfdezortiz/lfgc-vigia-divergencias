#!/usr/bin/env python3
"""Vigia de divergencias MACD + Fibonacci (operativa atribuida a David Trullas) -> avisos por Telegram.

Corre en la nube (GitHub Actions, revisando cada minuto) para avisar aunque el Mac este cerrado.
Mismas reglas que backtest/trullas_divergencias.py y tradingview/Divergencias_Fibo_Trullas.pine, en velas de 1, 5 y 15 min y 1 h:
  - Dos pivotes consecutivos (3 velas a cada lado): precio hace nuevo extremo y la linea MACD no.
  - Orden stop 1 tick tras el 23,6 % del tramo local; SL 1 tick tras el extremo; T1 38,2 %; T2 61,8 %.
  - Confirmaciones: MACD (siempre) + divergencia RSI(14) + menos volumen (en divisas no hay volumen).
  - La orden se cancela si se hace nuevo extremo antes de entrar o tras 60 velas.
  - Solo se avisa con el filtro completo: 3/3 (MACD + RSI + volumen); 2/2 en divisas (sin volumen). Sin avisos de «formándose»
    desde el 2026-11-04: solo la orden preparada y su seguimiento.
Activos (desde el 2026-11-05): solo XAUUSD, EURUSD, US100, DXY y BTCUSD (Nasdaq 100 con los precios del CFD de MT5,
solo en la sesión de contado de EE. UU., como en la auditoría; DXY y BTC con velas de Dukascopy, sin backtest).
Nasdaq 100 (US100), además: momentum de dos velas de Gian Luca en 5 min (momentum_gl.py, desde el 2026-11-10).
Oro y Nasdaq 100 (US100), además: modelo ICT barrido de liquidez + MSS + FVG en 5 y 15 min (fvg_ict.py), con el porqué
de cada orden.
Avisos: 🆕 orden preparada · ✅ activada · 🎯 T1 / T2 · ⛔ stop · ❌ cancelada / caducada · 📋 informe del dia.
Cada aviso lleva el resultado de la auditoria de ese activo (2023-2026, con costes).

Datos: Yahoo Finance (1m, 5m, 15m, 60m). Oro: contado de Dukascopy en tiempo real (desde 2026-11-02; Yahoo daba el futuro
GC=F con ~10 min de retraso); si Dukascopy falla, GC=F desplazado al precio contado de Swissquote (XAUUSD de MT5).
Uso: python3 vigia_divergencias.py [--prueba] [--informe] [--bucle MINUTOS]   (--prueba: muestra los avisos sin enviar ni guardar)
Telegram: variables TELEGRAM_TOKEN y TELEGRAM_CHAT, o config_telegram.json junto al script.
"""
import json, math, os, sys, time, urllib.request, urllib.parse, datetime as dt
import numpy as np, pandas as pd
import fvg_ict as FVG
import momentum_gl as GL

AQUI = os.path.dirname(os.path.abspath(__file__))
ESTADO = os.environ.get('VIGIA_ESTADO') or os.path.join(AQUI, 'data', 'vigia_divergencias.json')
UA = {'User-Agent': 'Mozilla/5.0'}
SQ = 'https://forex-data-feed.swissquote.com/public-quotes/bboquotes/instrument/XAU/USD'
CAPITAL, RIESGO = 820.0, 0.01
FONDEO, FONDEO_CAP = 'FTMO 10K', 10000.0                          # cuenta de fondeo: lotes y $ en cada aviso
FONDEO_RIESGO = float(os.environ.get('VIGIA_FONDEO_RIESGO', 0.015))   # 0,5 % en la nube: operación extra no validada
N, ESPERA, MIN_SEP, MAX_SEP = 3, 60, 5, 120
F_ENT, F_T1, F_T2 = 0.236, 0.382, 0.618
VENTANA = pd.Timedelta(hours=3)
AVISO_FORMACION = False                # 2026-11-04, Lucas: divergencias solo al preparar la orden y su seguimiento, sin «formándose»        # no se avisa de eventos mas antiguos (primer arranque, retrasos)
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

# Solo XAUUSD, EURUSD, US100, DXY y BTCUSD desde el 2026-11-05 (GBP/USD y USD/JPY fuera; el Nasdaq con precios del US100 de MT5).
# simbolo Yahoo, nombre, tick, decimales, confirmaciones minimas, tipo de tamano, veredicto de la auditoria
ACTIVOS = {
    'US100':    dict(duka='USATECH.IDX%2FUSD', yahoo='QQQ', nombre='Nasdaq 100', tick=0.1, dec=2, minconf=3, tipo='indice',
                     coste=1.5, apal=20, solo_rth=True,
                     audit='🧪 EN OBSERVACIÓN · Nasdaq M5 3/3 en sesión EE. UU.: PF 1,33 / 1,52 (52 ops 2023-26, auditado en QQQ). Prometedora, no validada.'),
    'XAUUSD':   dict(yahoo='GC=F', nombre='Oro contado', tick=0.01, dec=2, minconf=3, tipo='oro', coste=0.40, apal=20,
                     audit='⚠️ NO RENTABLE en el backtest del oro (M5 3/3: PF 0,59 / 1,41). Solo para seguir en demo.'),
    'EURUSD':   dict(yahoo='EURUSD=X', nombre='EUR/USD', tick=0.00001, dec=5, minconf=3, tipo='fx', coste=0.00012, apal=30,
                     audit='⚠️ NO RENTABLE en el backtest (M5: PF 0,45-0,89). Solo para seguir en demo.'),
    # DXY y BTC (desde el 2026-11-05): sin backtest; lotes cuando se conozca el tamaño del contrato en MT5 (valor = $ por punto y lote)
    'DXY':      dict(duka='DOLLAR.IDX%2FUSD', yahoo='DX-Y.NYB', nombre='Índice dólar', tick=0.001, dec=3, minconf=3, tipo='cfd',
                     valor=None, coste=0.03, apal=20,
                     audit='⚪ SIN BACKTEST: las divergencias no se han auditado en el DXY. Solo para seguir en demo.'),
    'BTCUSD':   dict(duka='BTC%2FUSD', yahoo='BTC-USD', nombre='Bitcoin', tick=0.1, dec=1, minconf=3, tipo='cfd',
                     valor=None, coste=30.0, apal=2,
                     audit='⚪ SIN BACKTEST: las divergencias no se han auditado en BTC. Solo para seguir en demo.'),
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


DUKA = ('https://freeserv.dukascopy.com/2.0/?path=chart/json3&instrument={ins}&offer_side=B&interval={iv}'
        '&splits=true&stocks=true&limit=1500&time_direction=P&timestamp={ts}&jsonp=_cb')
DUKA_IV = {'1m': '1MIN', '5m': '5MIN', '15m': '15MIN', '1h': '1HOUR'}


def velas_oro(tf):
    """Oro al contado en tiempo real (Dukascopy, la misma fuente que el backtest). Yahoo da el futuro GC con ~10 min
    de retraso, y en 1 min eso hacía llegar tarde los avisos."""
    return velas_duka(tf, 'XAU%2FUSD')


def velas_duka(tf, ins):
    """Velas cerradas de Dukascopy en tiempo real: oro (XAU%2FUSD) o Nasdaq 100 (USATECH.IDX%2FUSD, el US100 de MT5)."""
    mins = TFS[tf][0]
    ahora = pd.Timestamp.now(tz='UTC')
    req = urllib.request.Request(DUKA.format(ins=ins, iv=DUKA_IV[tf], ts=int(ahora.timestamp() * 1000)),
                                 headers=dict(UA, Referer='https://freeserv.dukascopy.com/'))
    for i in range(3):
        try:
            t = urllib.request.urlopen(req, timeout=20).read().decode()
            break
        except Exception:
            if i == 2:
                raise
            time.sleep(3 * (i + 1))
    a = json.loads(t[t.index('(') + 1:t.rindex(')')])
    d = pd.DataFrame(a, columns=['t', 'Open', 'High', 'Low', 'Close', 'Vol'])
    d.index = pd.to_datetime(d.pop('t'), unit='ms', utc=True)
    d = d.sort_index(); d = d[~d.index.duplicated(keep='last')]
    if len(d) < 100 or ahora - d.index[-1] > pd.Timedelta(minutes=2 * mins + 10) and mercado_abierto(ahora):
        raise ValueError('Dukascopy sin datos recientes')
    return d[d.index + pd.Timedelta(minutes=mins) <= ahora]       # solo velas cerradas


def tendencia_oro():
    """Contexto v2.3 del oro para la nota de fiabilidad: sesgo diario (cierre y EMA 20 frente a EMA 50 del ultimo dia
    cerrado) y tendencia horaria (EMA 20 frente a EMA 50 en 1 h). +1 alcista, -1 bajista, 0 neutral."""
    ahora = pd.Timestamp.now(tz='UTC')
    req = urllib.request.Request(DUKA.format(ins='XAU%2FUSD', iv='1DAY', ts=int(ahora.timestamp() * 1000)).replace('limit=1500', 'limit=300'),
                                 headers=dict(UA, Referer='https://freeserv.dukascopy.com/'))
    t = urllib.request.urlopen(req, timeout=20).read().decode()
    d = pd.DataFrame(json.loads(t[t.index('(') + 1:t.rindex(')')]), columns=['t', 'Open', 'High', 'Low', 'Close', 'Vol'])
    d.index = pd.to_datetime(d.pop('t'), unit='ms', utc=True)
    d = d.sort_index()
    d = d[(d.Vol > 0.2 * d.Vol.median()) & (d.index + pd.Timedelta(days=1) <= ahora)]   # sin domingos cortos ni el dia en curso
    e = lambda s, n: s.ewm(span=n, adjust=False).mean()
    c, f, s = d.Close.iloc[-1], e(d.Close, 20).iloc[-1], e(d.Close, 50).iloc[-1]
    h = velas_oro('1h')
    return dict(diario=1 if c > s and f > s else -1 if c < s and f < s else 0,
                horario=int(np.sign(e(h.Close, 20).iloc[-1] - e(h.Close, 50).iloc[-1])))


def nota_oro(tf, lado, R, ctx):
    """Nota de fiabilidad del oro: mezcla de la divergencia con el contexto de la v2.3 y el tamano del stop.
    Cifras de backtest/mezcla_divergencias_v23.py (3/3, objetivo T2, coste 0,40 $/oz, 2025-2026). Devuelve (icono, texto)."""
    if ctx is None:
        return '⚪', '🧭 Fiabilidad: sin contexto de tendencia ahora mismo (fallo de datos).'
    d, h = ctx['diario'] * lado, ctx['horario'] * lado
    contra = d < 0 and h < 0
    if tf == '1m':
        if R >= 2 and not contra:
            g, por = '🟢 ALTA', 'PF 2,4 en 37 ops (1 min, SL ≥ 2 $ y sin ir contra diario y horario a la vez)'
        elif R >= 1.5:
            g, por = '🟡 MEDIA', 'PF 1,5–1,9 (1 min con SL ≥ 1,5 $)'
        else:
            g, por = '🔴 BAJA', 'PF < 1 en 125 ops (1 min con SL < 1,5 $: el coste se come la ventaja)'
    elif tf == '5m':
        if R < 1:
            g, por = '🔴 BAJA', 'stop < 1 $: el coste se come la ventaja'
        elif not contra:
            g, por = '🟢 ALTA', 'PF 2,1 en 21 ops (5 min sin ir contra diario y horario a la vez)'
        else:
            g, por = '🟡 MEDIA', 'PF 1,0 en 25 ops (5 min contra el diario y el horario)'
    else:
        g, por = '🟡 MEDIA', 'pocas señales en 15 min y 1 h (PF 1,1–1,5, sin validar)'
    ico = lambda x: '✅ a favor' if x > 0 else '❌ en contra' if x < 0 else '➖ neutral'
    return g[0], (f"🧭 <b>Fiabilidad {g}</b> · diario {ico(d)} · horario {ico(h)} · SL {R:.2f} $\n"
                  f"   Histórico 2025-26 de este tipo: {por}. Muestra pequeña: extra al 0,5 %, decides tú.")


# Modelo ICT (barrido + MSS + FVG, fvg_ict.py) en el oro y en el Nasdaq 100: resultado del backtest/fvg_ict_backtest.py
# (2023-2026, killzones, objetivo 2R). No validado en ninguno de los dos: los avisos van marcados como tales.
_NV = '🧭 <b>Fiabilidad 🔴 NO VALIDADA</b> · backtest {} 2023-26: {}. Recomendación: solo demo hasta tener muestra real.'
FVG_ACT = {
    'XAUUSD': dict(ins='XAU%2FUSD', buf=0.50, coste=0.40, ud='$', valor=100.0, apal=20, nota={
        '5m': _NV.format('oro', '38 ops, acierto 29 %, PF 0,86 / 0,50 (−7,9R), sin ventaja ni antes de costes'),
        '15m': _NV.format('oro', '43 ops, acierto 33 %, PF 0,87 / 0,51 (−7,6R), sin ventaja ni antes de costes')}),
    # US100.cash de la FTMO: tamaño del contrato 1 → 1 $ por punto y lote (especificación de MT5 de Lucas, 2026-11-04)
    'US100': dict(ins='USATECH.IDX%2FUSD', buf=3.0, coste=1.5, ud='pt', valor=1.0, apal=20, nota={
        '5m': _NV.format('Nasdaq', '32 ops, acierto 38 %, PF 0,77 / 1,51 (−1,1R): pierde en 2023-24 y gana en 2025-26'),
        '15m': _NV.format('Nasdaq', '51 ops, acierto 35 %, PF 0,83 / 0,84 (−5,5R)')}),
}


def fondeo_fvg(sim, s, rr=None):
    """Lotes de la FTMO al riesgo de las extras: riesgo $ / (stop × valor de 1 punto por lote), con tope de apalancamiento."""
    c = FVG_ACT[sim]
    lotes = math.floor(FONDEO_CAP * FONDEO_RIESGO / (s['R'] * c['valor']) * 100) / 100
    tope = math.floor(FONDEO_CAP * c['apal'] / (c['valor'] * s['ent']) * 100) / 100
    nota = ' (tope de apalancamiento)' if lotes > tope else ''
    lotes = max(0.01, min(lotes, tope)); sl = lotes * c['valor'] * s['R']
    return (f"Lotes FTMO ({FONDEO_RIESGO:.1%}): <b>{lotes:.2f}</b>{nota} → SL <b>−{sl:.0f} $</b> · "
            f"TP <b>+{sl * (rr or FVG.BASE['rr']):.0f} $</b>")


def fvg_avisos(sim, d, tf, mins, tfl, ahora, enviados, salida, hoy):
    """Avisos del modelo FVG (5 y 15 min): barrido de liquidez (aviso previo) y la orden con su porqué."""
    c = FVG_ACT[sim]
    pre = 'fvg' if sim == 'XAUUSD' else 'fvg|'          # el oro conserva sus claves (no repetir avisos ya enviados)
    setups, barridos = FVG.escanear(d, mins, dict(ventanas='kz', buf=c['buf']))
    for b in barridos:                                # solo los que pueden acabar en una orden dentro de una killzone
        clave = f"{sim}|{pre}{tf}|{b['id']}"
        if clave in enviados or ahora - b['t'] > VENTANA:
            continue
        if FVG._ventana(b['t'], 'kz') or FVG._ventana(b['limite'], 'kz'):
            salida.append((b['t'], FVG.mensaje(b, 'barrido', b['t'], tfl, sim=sim, coste=c['coste'], ud=c['ud'])))
        enviados.add(clave)
    for s in setups:
        for t, evento, precio in s['ev']:
            clave = f"{sim}|{pre}{tf}|{s['id']}|{evento}"
            if t.date() == ahora.date():
                hoy.append((sim, s, evento, f'FVG {tfl}'))
            if clave in enviados or ahora - t > VENTANA:
                continue
            salida.append((t, FVG.mensaje(s, evento, t, tfl, fondeo_fvg(sim, s), c['nota'][tf], sim, c['coste'], c['ud'])))
            enviados.add(clave)


GL_NOTA = ('🧭 Fiabilidad 🧪 EN OBSERVACIÓN · backtest US100 2023-26: 562 ops, acierto 47 %, E +0,09R neta, '
           'PF 1,20 / 1,17 (OPERATIVA §27). Extra al 0,5 %, decides tú.')


def gl_avisos(d, ahora, enviados, salida, hoy):
    """Momentum de dos velas de Gian Luca (US100, 5 min): entrada a mercado, TP, stop y cierre de sesión."""
    c = FVG_ACT['US100']
    for s in GL.escanear(d, 5):
        for t, evento, precio in s['ev']:
            clave = f"US100|gl|{s['id']}|{evento}"
            if t.date() == ahora.date():
                hoy.append(('US100', s, evento, 'GL 5 min'))
            if clave in enviados or ahora - t > VENTANA:
                continue
            salida.append((t, GL.mensaje(s, evento, t, '5 min', fondeo_fvg('US100', s, GL.BASE['rr']), GL_NOTA,
                                         'US100', c['coste'], c['ud'])))
            enviados.add(clave)


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
            out.append((f'forma|{clave}', lado, 0.236 * tramo + a['tick'], '\n'.join([
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
    if a['tipo'] == 'cfd':
        return 'lotes: falta el tamaño del contrato en MT5'
    if a['tipo'] == 'indice':                                 # US100: 1 $ por punto y lote
        k = max(1, min(int(obj / R * 100), int(CAPITAL * a['apal'] / precio * 100)))
        return f"{k / 100:.2f} lotes → riesgo ≈ {k * R / 100:.0f} $ ({k * R / 100 / CAPITAL:.1%})"
    if a['tipo'] == 'oro':                                    # 0,01 lotes = 1 oz
        usd_ud, ud_max = R, CAPITAL * a['apal'] / precio
    else:                                                     # 0,01 lotes = 1.000 unidades
        usd_ud = R * 1000 / (precio if a['yahoo'].startswith('USD') else 1)
        ud_max = CAPITAL * a['apal'] / (1000 * (1 if a['yahoo'].startswith('USD') else precio))
    k = max(1, min(int(obj // usd_ud), int(ud_max)))
    tope = ' (tope por apalancamiento)' if int(obj // usd_ud) > int(ud_max) else ''
    return f"{k / 100:.2f} lotes → riesgo ≈ {k * usd_ud:.0f} $ ({k * usd_ud / CAPITAL:.1%}){tope}"


def fondeo_calc(a, s):
    """Lotes y dólares en la cuenta de fondeo (CFD: 1 lote oro = 100 oz, 1 lote divisas = 100.000 unidades,
    1 lote US100.cash = 1 $ por punto). Devuelve (lotes, SL $, T1 $, T2 $, tope), o None si falta el tamaño del contrato."""
    if a['tipo'] == 'cfd':
        if not a.get('valor'):
            return None
        usd_lote, nocional = s['R'] * a['valor'], s['ent'] * a['valor']
    elif a['tipo'] == 'indice':
        usd_lote, nocional = s['R'], s['ent']
    elif a['tipo'] == 'oro':
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
    return lotes, sl, g(s['t1']), g(s['t2']), bool(nota)


def fondeo(a, s):
    f = fondeo_calc(a, s)
    if f is None:
        return f"🏦 {FONDEO}: falta el tamaño del contrato en MT5 para calcular los lotes."
    lotes, sl, t1, t2, tope = f
    return (f"🏦 <b>{FONDEO}</b> ({FONDEO_RIESGO:.1%}): <b>{lotes:.2f} lotes</b>{' (tope de apalancamiento)' if tope else ''} · "
            f"SL <b>−{sl:.0f} $</b> · T1 <b>+{t1:.0f} $</b> · T2 <b>+{t2:.0f} $</b>")


def fmt(a, x):
    return f"{x:.{a['dec']}f}"


def mensaje(sim, a, s, evento, t, precio, tfl='5 min', audit=None, nota=None):
    lado = 'VENTA' if s['lado'] < 0 else 'COMPRA'
    hora = t.tz_convert('Europe/Madrid').strftime('%H:%M')
    rr = lambda x: abs(x - s['ent']) / s['R'] if s['R'] else 0
    if evento == 'nueva':
        compra = s['lado'] > 0
        f = fondeo_calc(a, s)
        u = {'oro': ' $', 'indice': ' pt', 'cfd': ' pt'}.get(a['tipo'], '')
        if f is None:
            lotes = [f"Lotes: falta el tamaño del contrato de {sim} en MT5; arriesga {FONDEO_RIESGO:.1%} ({FONDEO_CAP * FONDEO_RIESGO:.0f} $) "
                     f"con lotes = {FONDEO_CAP * FONDEO_RIESGO:.0f} $ ÷ ({fmt(a, s['R'])} × $ por punto y lote)."]
        else:
            lt, sl_usd, t1_usd, t2_usd, tope = f
            lotes = [f"Lotes FTMO ({FONDEO_RIESGO:.1%}): <b>{lt:.2f}</b>{' (tope de apalancamiento)' if tope else ''} → "
                     f"SL <b>−{sl_usd:.0f} $</b> · TP <b>+{t2_usd:.0f} $</b>"]
        avisos = []
        if tfl == '1 min':
            avisos.append('1 min va fuera del plan de fondeo: en conjunto no es rentable; mira la fiabilidad.')
        if a['coste'] > 0.25 * s['R']:
            avisos.append(f"Stop muy corto: el spread (~{fmt(a, a['coste'])}{u}) es el {a['coste'] / s['R']:.0%} del riesgo.")
        return '\n'.join([
            f"🆕 {'🟢' if compra else '🔴'} <b>{lado} PREPARADA · {sim} · {tfl}</b> · {hora}",
            f"Divergencia {'alcista' if compra else 'bajista'} (Trullás) · confirmaciones {s['conf']}/{s['maxconf']}"
            + (' ⭐' if s['conf'] == s['maxconf'] else ' (filtro incompleto)'),
            *([nota] if nota else []),
            SEP,
            '📋 <b>ORDEN PARA MT5</b>',
            f"Tipo: <b>{'buy stop' if compra else 'sell stop'}</b>",
            f"Entrada: <code>{fmt(a, s['ent'])}</code>",
            f"SL: <code>{fmt(a, s['sl'])}</code> ({fmt(a, s['R'])}{u})",
            f"TP: <code>{fmt(a, s['t2'])}</code> ({rr(s['t2']):.1f}R) · parcial opcional en <code>{fmt(a, s['t1'])}</code> ({rr(s['t1']):.1f}R)",
            *lotes,
            SEP,
            *[f"⚠️ {x}" for x in avisos],
            f"⏳ Cancela la orden si hace nuevo {'mínimo' if compra else 'máximo'} (<code>{fmt(a, s['ext'])}</code>) antes de entrar o tras 60 velas.",
            f"<i>{audit or a['audit']}</i>"])
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
    copia_github(texto)                               # todos los activos (desde el 2026-11-05), no solo el oro


def copia_github(texto):
    """En la nube: copia cada aviso como comentario de la issue AVISOS_ISSUE (la lee Claude)."""
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
    notas = estado.setdefault('notas', {})            # icono de fiabilidad de cada divergencia del oro (para el informe)
    ctx = []

    def contexto():                                   # tendencias v2.3 del oro, solo si hay aviso que puntuar
        if not ctx:
            try:
                ctx.append(tendencia_oro())
            except Exception as ex:
                ctx.append(None); errores.append(f'contexto del oro no disponible ({ex})')
        return ctx[0]
    for tf, (mins, _, _, tfl, previos, audit_tf) in TFS.items():
        # cada temporalidad se descarga justo despues de cerrar su vela (y siempre en la primera vuelta y el informe)
        if not (todas or toca_informe or minuto % mins == 0):
            continue
        for sim, a in ACTIVOS.items():
            try:
                d = None
                if sim == 'XAUUSD':                           # contado en tiempo real; si falla, futuro GC desplazado
                    try:
                        d = velas_oro(tf)
                    except Exception as ex:
                        errores.append(f'oro {tf}: Dukascopy no disponible ({ex}); se usa Yahoo con ~10 min de retraso')
                if d is None and 'duka' in a:              # US100: CFD de Dukascopy en tiempo real (los niveles de MT5)
                    try:
                        d = velas_duka(tf, a['duka'])
                    except ValueError:
                        continue                              # pausa diaria del mercado: sin velas nuevas
                    if a.get('solo_rth'):                     # como la auditoría en QQQ: solo la sesión de contado de EE. UU.
                        ny = d.index.tz_convert('America/New_York'); x = ny.hour + ny.minute / 60
                        d = d[(x >= 9.5) & (x < 16) & (ny.weekday < 5)]
                if d is None:
                    d = velas(a['yahoo'], tf)
                    if sim == 'XAUUSD':                       # futuro GC -> precio contado (lo que ves en MT5)
                        try:
                            if base_oro is None:
                                base_oro = oro_contado() - d.Close.iloc[-1]
                            d[['Open', 'High', 'Low', 'Close']] += base_oro
                        except Exception:
                            errores.append('oro contado no disponible: niveles del futuro GC')
                di = indicadores(d)
                audit = audit_tf or a['audit']
                if sim == 'XAUUSD' and tf in ('5m', '15m'):     # modelo ICT: barrido + MSS + FVG
                    try:
                        fvg_avisos('XAUUSD', d, tf, mins, tfl, ahora, enviados, salida, hoy)
                    except Exception as ex:
                        errores.append(f'FVG {tf}: {ex}')
                pre = '' if tf == '5m' else f'{tf}|'          # 5 min conserva las claves antiguas (no repetir avisos)
                aa = a                                          # todas: solo filtro completo (minconf=3)
                if AVISO_FORMACION and (previos or sim == 'XAUUSD'):   # aviso previo «formándose» (desactivado)
                    for clave, lado, R, texto in vigilancia(di, a, mins):
                        clave = f'{sim}|{pre}{clave}'
                        if clave not in enviados:
                            nota = '\n' + nota_oro(tf, lado, R, contexto())[1] if sim == 'XAUUSD' else ''
                            salida.append((ahora, texto.replace('</b> · 5 min', f"</b> · {sim} ({a['nombre']}) · {tfl}", 1)
                                           + nota + '\n' + audit))
                            enviados.add(clave)
                for s in setups(di, aa, mins):
                    for t, evento, precio in s['ev']:
                        clave = f"{sim}|{pre}{s['id']}|{evento}"
                        if t.date() == ahora.date():
                            hoy.append((sim, s, evento, tfl))
                        if clave in enviados or ahora - t > VENTANA:
                            continue
                        nota = None
                        if sim == 'XAUUSD' and evento == 'nueva':    # nota de fiabilidad (mezcla con la v2.3)
                            ico, nota = nota_oro(tf, s['lado'], s['R'], contexto())
                            notas[f"{pre}{s['id']}"] = ico
                        salida.append((t, mensaje(sim, a, s, evento, t, precio, tfl, audit, nota)))
                        enviados.add(clave)
            except Exception as e:
                errores.append(f'{sim} {tf}: {e}')
    for tf in ('5m', '15m'):                          # modelo FVG en el Nasdaq 100 (US100 de MT5, Dukascopy en tiempo real)
        mins, tfl = TFS[tf][0], TFS[tf][3]
        if todas or toca_informe or minuto % mins == 0:
            try:
                d_us = velas_duka(tf, FVG_ACT['US100']['ins'])
            except ValueError:
                continue                              # pausa diaria del índice: sin velas nuevas
            except Exception as ex:
                errores.append(f'US100 {tf}: {ex}'); continue
            try:
                fvg_avisos('US100', d_us, tf, mins, tfl, ahora, enviados, salida, hoy)
            except Exception as ex:
                errores.append(f'FVG US100 {tf}: {ex}')
            if tf == '5m':                            # momentum de dos velas de Gian Luca (desde el 2026-11-10)
                try:
                    gl_avisos(d_us, ahora, enviados, salida, hoy)
                except Exception as ex:
                    errores.append(f'Momentum GL US100: {ex}')
    salida.sort(key=lambda x: x[0])
    mensajes = [m for _, m in salida]
    if toca_informe:
        mensajes.append(texto_informe(hoy, errores, notas))
        estado['informe'] = str(madrid.date())
    if prueba:
        print('\n\n'.join(mensajes) or 'Sin avisos nuevos.')
        print('Errores:', errores) if errores else None
        return
    for m in mensajes:
        tg(m)
        print('ENVIADO', ahora.strftime('%Y-%m-%d %H:%M'), m.split(chr(10))[0][:120], flush=True)
    estado['enviados'] = sorted(enviados)[-3000:]
    estado['notas'] = dict(sorted(notas.items(), key=lambda x: x[0][-12:])[-500:])
    os.makedirs(os.path.dirname(ESTADO), exist_ok=True)
    json.dump(estado, open(ESTADO, 'w'))
    if errores:
        print('Errores:', errores)


def texto_informe(hoy, errores, notas=None):
    lineas = [f"📋 <b>INFORME DIVERGENCIAS · {dt.date.today():%d/%m}</b>", SEP]
    finales, notas = {}, notas or {}
    pre = {'1 min': '1m|', '5 min': '', '15 min': '15m|', '1 h': '1h|'}
    for sim, s, evento, tfl in hoy:
        finales[(sim, tfl, s['id'])] = (sim, s, evento, tfl)
    if not finales:
        lineas.append('Hoy no hubo divergencias que cumplan las reglas.')
    total, por_nota, fvg, gl = 0.0, {}, [0, 0.0], [0, 0.0]
    for sim, s, evento, tfl in sorted(finales.values(), key=lambda x: (x[3], x[0])):
        ico = notas.get(f"{pre.get(tfl, '')}{s['id']}", '') if sim == 'XAUUSD' else ''
        r = {'t2': abs(s['t2'] - s['ent']) / s['R'], 'stop': -1.0, 'cierre': s.get('r_cierre')}.get(evento)
        if r is not None:
            total += r
            if tfl.startswith('FVG'):                 # oro y Nasdaq juntos
                fvg[0] += 1; fvg[1] += r
            elif tfl.startswith('GL'):
                gl[0] += 1; gl[1] += r
            elif ico:
                n, x = por_nota.get(ico, (0, 0.0)); por_nota[ico] = (n + 1, x + r)
        nom = {'nueva': 'pendiente', 'activada': 'abierta', 't1': 'abierta (T1 tocado)', 't2': 'T2 ✅',
               'stop': 'stop ⛔', 'cancelada': 'cancelada', 'caducada': 'caducada', 'cierre': 'cerrada a las 16:00 NY'}[evento]
        conf = f" {s['conf']}/{s['maxconf']}" if 'conf' in s else s.get('etq', ' (barrido + MSS)')
        lineas.append(f"• {sim} {tfl} {'venta' if s['lado'] < 0 else 'compra'}{conf}{' ' + ico if ico else ''}: {nom}")
    if finales:
        lineas.append(f"\nResultado de las cerradas hoy (objetivo T2): {total:+.1f}R")
    if fvg[0]:
        lineas.append(f"Modelo FVG (ICT, no validado): {fvg[0]} ops {fvg[1]:+.1f}R")
    if gl[0]:
        lineas.append(f"Momentum GL (US100, en observación): {gl[0]} ops {gl[1]:+.1f}R")
    if por_nota:                                      # feedback de la mezcla: ¿las 🟢 rinden mejor que las 🔴?
        lineas.append('Oro por fiabilidad: ' + ' · '.join(f"{k} {n} ops {x:+.1f}R" for k, (n, x) in
                                                            sorted(por_nota.items(), key=lambda z: '🟢🟡🔴⚪'.find(z[0]))))
    lineas.append('\nRecuerda: solo el Nasdaq (US100, 5 min, 3/3, sesión EE. UU.) pasó la auditoría, y con muestra pequeña; las señales con menos confirmaciones son menos fiables.')
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
