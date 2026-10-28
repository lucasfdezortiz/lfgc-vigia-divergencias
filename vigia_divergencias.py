#!/usr/bin/env python3
"""Vigia de divergencias MACD + Fibonacci (operativa atribuida a David Trullas) -> avisos por Telegram.

Corre en la nube (GitHub Actions, cada 15 min) para avisar aunque el Mac este cerrado.
Mismas reglas que backtest/trullas_divergencias.py y tradingview/Divergencias_Fibo_Trullas.pine, en velas de 5 min:
  - Dos pivotes consecutivos (3 velas a cada lado): precio hace nuevo extremo y la linea MACD no.
  - Orden stop 1 tick tras el 23,6 % del tramo local; SL 1 tick tras el extremo; T1 38,2 %; T2 61,8 %.
  - Confirmaciones: MACD (siempre) + divergencia RSI(14) + menos volumen (en divisas no hay volumen).
  - La orden se cancela si se hace nuevo extremo antes de entrar o tras 60 velas (5 h).
Avisos: 🆕 orden preparada · ✅ activada · 🎯 T1 / T2 · ⛔ stop · ❌ cancelada / caducada · 📋 informe del dia.
Cada aviso lleva el resultado de la auditoria de ese activo (2023-2026, con costes).

Datos: Yahoo Finance 5 min. Oro: futuro GC=F desplazado al precio contado de Swissquote (XAUUSD de MT5).
Uso: python3 vigia_divergencias.py [--prueba] [--informe]   (--prueba: muestra los avisos sin enviar ni guardar)
Telegram: variables TELEGRAM_TOKEN y TELEGRAM_CHAT, o config_telegram.json junto al script.
"""
import json, os, sys, time, urllib.request, urllib.parse, datetime as dt
import numpy as np, pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
ESTADO = os.environ.get('VIGIA_ESTADO') or os.path.join(AQUI, 'data', 'vigia_divergencias.json')
UA = {'User-Agent': 'Mozilla/5.0'}
SQ = 'https://forex-data-feed.swissquote.com/public-quotes/bboquotes/instrument/XAU/USD'
CAPITAL, RIESGO = 820.0, 0.01
N, ESPERA, MIN_SEP, MAX_SEP = 3, 60, 5, 120
F_ENT, F_T1, F_T2 = 0.236, 0.382, 0.618
VENTANA = pd.Timedelta(hours=3)        # no se avisa de eventos mas antiguos (primer arranque, retrasos)
SEP = '━━━━━━━━━━━━━━━━━━'

# simbolo Yahoo, nombre, tick, decimales, confirmaciones minimas, tipo de tamano, veredicto de la auditoria
ACTIVOS = {
    'QQQ':      dict(yahoo='QQQ', nombre='Nasdaq 100', tick=0.01, dec=2, minconf=3, tipo='accion', coste=0.03, apal=5,
                     audit='🧪 EN OBSERVACIÓN · Nasdaq M5 3/3 en sesión EE. UU.: PF 1,33 / 1,52 (52 ops 2023-26). Prometedora, no validada.'),
    'XAUUSD':   dict(yahoo='GC=F', nombre='Oro contado', tick=0.01, dec=2, minconf=3, tipo='oro', coste=0.40, apal=20,
                     audit='⚠️ NO RENTABLE en el backtest del oro (M5 3/3: PF 0,59 / 1,41). Solo para seguir en demo.'),
    'EURUSD':   dict(yahoo='EURUSD=X', nombre='EUR/USD', tick=0.00001, dec=5, minconf=2, tipo='fx', coste=0.00012, apal=30,
                     audit='⚠️ NO RENTABLE en el backtest (M5: PF 0,45-0,89). Solo para seguir en demo.'),
    'GBPUSD':   dict(yahoo='GBPUSD=X', nombre='GBP/USD', tick=0.00001, dec=5, minconf=2, tipo='fx', coste=0.00015, apal=30,
                     audit='⚠️ NO VALIDADA en el backtest (M5: PF 0,47-1,26). Solo para seguir en demo.'),
    'USDJPY':   dict(yahoo='USDJPY=X', nombre='USD/JPY', tick=0.001, dec=3, minconf=2, tipo='fx', coste=0.015, apal=30,
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


def velas_5m(yahoo):
    r = json.loads(_get(f'https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(yahoo)}'
                        '?interval=5m&range=5d'))['chart']['result'][0]
    q = r['indicators']['quote'][0]
    d = pd.DataFrame({'Open': q['open'], 'High': q['high'], 'Low': q['low'], 'Close': q['close'],
                      'Vol': q['volume']}, index=pd.to_datetime(r['timestamp'], unit='s', utc=True))
    d.index = d.index.floor('5min')
    d = d[~d.index.duplicated(keep='last')].dropna(subset=['Open', 'High', 'Low', 'Close'])
    d['Vol'] = d.Vol.fillna(0)
    ahora = pd.Timestamp.now(tz='UTC')
    return d[d.index + pd.Timedelta(minutes=5) <= ahora]          # solo velas cerradas


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


def setups(d, a):
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
            t0 = idx[ini - 1] + pd.Timedelta(minutes=5)      # cierre de la vela que confirma el pivote
            if (C[ini - 1] <= s['ent']) if lado < 0 else (C[ini - 1] >= s['ent']):
                continue                                       # ya habia pasado el nivel: no se persigue
            ev = [(t0, 'nueva', None)]
            j, estado = ini, 'pendiente'
            while j < n and estado not in ('stop', 't2', 'cancelada', 'caducada'):
                hi, lo, tj = H[j], L[j], idx[j] + pd.Timedelta(minutes=5)
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


def fmt(a, x):
    return f"{x:.{a['dec']}f}"


def mensaje(sim, a, s, evento, t, precio):
    lado = 'VENTA' if s['lado'] < 0 else 'COMPRA'
    hora = t.tz_convert('Europe/Madrid').strftime('%H:%M')
    rr = lambda x: abs(x - s['ent']) / s['R'] if s['R'] else 0
    if evento == 'nueva':
        return '\n'.join([
            f"🆕 <b>{'🔴' if s['lado'] < 0 else '🟢'} {lado} PREPARADA · {sim}</b> ({a['nombre']}) · 5 min · {hora}",
            f"Divergencia {'bajista' if s['lado'] < 0 else 'alcista'} MACD · confirmaciones {s['conf']}/{s['maxconf']}",
            SEP,
            f"👉 Orden {'sell stop' if s['lado'] < 0 else 'buy stop'} en <code>{fmt(a, s['ent'])}</code> (23,6 %)",
            f"⛔ SL: <code>{fmt(a, s['sl'])}</code> (riesgo {fmt(a, s['R'])})",
            f"🎯 T1 38,2 %: <code>{fmt(a, s['t1'])}</code> ({rr(s['t1']):.1f}R)",
            f"🎯 T2 61,8 %: <code>{fmt(a, s['t2'])}</code> ({rr(s['t2']):.1f}R)",
            f"📦 Tamaño (1 % de {CAPITAL:.0f} $): {tamano(a, s['R'], s['ent'])}",
            *([f"⚠️ Stop muy corto: el spread (~{fmt(a, a['coste'])}) es el {a['coste'] / s['R']:.0%} del riesgo."]
              if a['coste'] > 0.25 * s['R'] else []),
            f"⏳ Cancelar si hace nuevo {'máximo' if s['lado'] < 0 else 'mínimo'} ({fmt(a, s['ext'])}) o a las 5 h sin entrar.",
            SEP, a['audit']])
    textos = {
        'activada': f"✅ <b>{lado} ACTIVADA · {sim}</b> en <code>{fmt(a, s['ent'])}</code> ({hora}). SL <code>{fmt(a, s['sl'])}</code> · T1 <code>{fmt(a, s['t1'])}</code> · T2 <code>{fmt(a, s['t2'])}</code>",
        't1': f"🎯 <b>T1 · {sim}</b> {lado.lower()}: tocado <code>{fmt(a, s['t1'])}</code> (+{rr(s['t1']):.1f}R) a las {hora}.",
        't2': f"🎯🎯 <b>T2 · {sim}</b> {lado.lower()}: tocado <code>{fmt(a, s['t2'])}</code> (+{rr(s['t2']):.1f}R) a las {hora}. Operación terminada.",
        'stop': f"⛔ <b>STOP · {sim}</b> {lado.lower()}: tocado <code>{fmt(a, s['sl'])}</code> (−1R) a las {hora}.",
        'cancelada': f"❌ <b>CANCELADA · {sim}</b> {lado.lower()} {fmt(a, s['ent'])}: nuevo extremo antes de entrar. Quita la orden.",
        'caducada': f"⌛ <b>CADUCADA · {sim}</b> {lado.lower()} {fmt(a, s['ent'])}: 5 h sin activarse. Quita la orden.",
    }
    return textos[evento]


# ---------------------------------------------------------------- telegram y estado
def tg(texto):
    tok, chat = os.environ.get('TELEGRAM_TOKEN'), os.environ.get('TELEGRAM_CHAT')
    if not tok:
        c = json.load(open(os.path.join(AQUI, 'config_telegram.json')))
        tok, chat = c['token'], c['chat_id']
    datos = urllib.parse.urlencode({'chat_id': chat, 'text': texto, 'parse_mode': 'HTML'}).encode()
    urllib.request.urlopen(f'https://api.telegram.org/bot{tok}/sendMessage', data=datos, timeout=30).read()


def cargar_estado():
    try:
        return json.load(open(ESTADO))
    except Exception:
        return {'enviados': [], 'informe': ''}


def main(prueba=False, informe=False):
    estado = cargar_estado(); enviados = set(estado['enviados'])
    ahora = pd.Timestamp.now(tz='UTC'); salida, hoy, errores = [], [], []
    for sim, a in ACTIVOS.items():
        try:
            d = velas_5m(a['yahoo'])
            if sim == 'XAUUSD':                               # futuro GC -> precio contado (lo que ves en MT5)
                try:
                    base = oro_contado() - d.Close.iloc[-1]
                    d[['Open', 'High', 'Low', 'Close']] += base
                except Exception:
                    errores.append('oro contado no disponible: niveles del futuro GC')
            for s in setups(indicadores(d), a):
                for t, evento, precio in s['ev']:
                    clave = f"{sim}|{s['id']}|{evento}"
                    if t.date() == ahora.date():
                        hoy.append((sim, s, evento))
                    if clave in enviados or ahora - t > VENTANA:
                        continue
                    salida.append((t, mensaje(sim, a, s, evento, t, precio)))
                    enviados.add(clave)
        except Exception as e:
            errores.append(f'{sim}: {e}')
    salida.sort(key=lambda x: x[0])
    mensajes = [m for _, m in salida]
    madrid = ahora.tz_convert('Europe/Madrid')
    if informe or (madrid.hour >= 22 and estado.get('informe') != str(madrid.date()) and madrid.weekday() < 5):
        mensajes.append(texto_informe(hoy, errores))
        estado['informe'] = str(madrid.date())
    if prueba:
        print('\n\n'.join(mensajes) or 'Sin avisos nuevos.')
        print('Errores:', errores) if errores else None
        return
    for m in mensajes:
        tg(m)
    estado['enviados'] = sorted(enviados)[-3000:]
    os.makedirs(os.path.dirname(ESTADO), exist_ok=True)
    json.dump(estado, open(ESTADO, 'w'))
    if errores:
        print('Errores:', errores)


def texto_informe(hoy, errores):
    lineas = [f"📋 <b>INFORME DIVERGENCIAS · {dt.date.today():%d/%m}</b>", SEP]
    finales = {}
    for sim, s, evento in hoy:
        finales[(sim, s['id'])] = (sim, s, evento)
    if not finales:
        lineas.append('Hoy no hubo divergencias que cumplan las reglas.')
    total = 0.0
    for sim, s, evento in finales.values():
        r = {'t2': abs(s['t2'] - s['ent']) / s['R'], 'stop': -1.0}.get(evento)
        if r is not None:
            total += r
        nom = {'nueva': 'pendiente', 'activada': 'abierta', 't1': 'abierta (T1 tocado)', 't2': 'T2 ✅',
               'stop': 'stop ⛔', 'cancelada': 'cancelada', 'caducada': 'caducada'}[evento]
        lineas.append(f"• {sim} {'venta' if s['lado'] < 0 else 'compra'} {s['conf']}/{s['maxconf']}: {nom}")
    if finales:
        lineas.append(f"\nResultado de las cerradas hoy (objetivo T2): {total:+.1f}R")
    lineas.append('\nRecuerda: solo QQQ (5 min, 3/3, sesión EE. UU.) pasó la auditoría, y con muestra pequeña.')
    if errores:
        lineas.append('⚠️ ' + '; '.join(errores))
    return '\n'.join(lineas)


if __name__ == '__main__':
    main(prueba='--prueba' in sys.argv, informe='--informe' in sys.argv)
