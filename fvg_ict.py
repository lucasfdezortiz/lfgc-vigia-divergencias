#!/usr/bin/env python3
"""Modelo ICT del oro: Liquidity Sweep + Market Structure Shift (MSS) + Fair Value Gap (FVG).

Reglas objetivas a partir del PDF «Significado de FVG en trading» (conversación de Lucas con ChatGPT, 2026-11-04).
Las usan a la vez el backtest (backtest/fvg_ict_backtest.py) y el vigía en la nube (vigia_divergencias.py), así que
lo que se avisa en directo es exactamente lo que se ha medido. Compra (la venta es simétrica):
  1. Liquidez: mínimo del día anterior (PDL), mínimo de Asia (20:00-00:00 Nueva York) y mínimo de Londres
     (02:00-05:00 Nueva York). Un nivel se gasta la primera vez que el precio lo perfora.
  2. Liquidity Sweep: una vela perfora el nivel con la mecha y cierra por encima («barre y vuelve»).
     Si una vela cierra por debajo del nivel antes del MSS, era una ruptura, no un barrido: se anula.
  3. MSS: en las M velas siguientes, una vela cierra por encima del último máximo de swing (pivote de 2 velas)
     anterior al barrido.
  4. FVG: el desplazamiento que produce el MSS deja un hueco de 3 velas (mínimo de la 3.ª > máximo de la 1.ª);
     se usa el último, de tamaño mínimo P['minfvg'] × ATR. Sin FVG no hay operación.
  5. Entrada: orden limitada en el 50 % del FVG (Consequent Encroachment). Válida W velas; se cancela si el precio
     llega al objetivo sin haber entrado.
  6. Stop: detrás del extremo del barrido (el swing que invalida la idea) más un margen. Objetivo: 2R.
  7. Gestión de cuenta fondeada: una operación a la vez, máximo 3 al día, y se para el día tras 2 pérdidas seguidas.
     Si sigue abierta a las 16:00 de Nueva York, se cierra.
Filtros opcionales: ventana horaria (killzones o Silver Bullet, en hora de Nueva York para que el cambio de
hora no la mueva) y tendencia de 1 h a favor (EMA 20 frente a EMA 50).
"""
import math
import numpy as np, pandas as pd

NY = 'America/New_York'
KILLZONES = {'Londres': (2, 5), 'Nueva York AM': (7, 10), 'Nueva York PM': (12, 14)}        # hora de Nueva York
SILVER_BULLET = {'Londres': (3, 4), 'Nueva York AM': (10, 11), 'Nueva York PM': (14, 15)}
BASE = dict(N=2, M=12, W=12, minfvg=0.10, buf=0.50, rr=2.0, entrada='ce', ventanas=None, sesgo=False,
            max_dia=3, max_perdidas=2, cierre_ny=16, minR=0.0)


def _atr(d, n=14):
    tr = pd.concat([d.High - d.Low, (d.High - d.Close.shift()).abs(), (d.Low - d.Close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean().values


def _niveles(d):
    """Por vela, los niveles de liquidez vigentes: dict nombre -> (precio, lado) con lado -1 = máximo (BSL), +1 = mínimo (SSL)."""
    ny = d.index.tz_convert(NY)
    dia = (ny + pd.Timedelta(hours=7)).normalize().tz_localize(None)        # el día de mercado empieza a las 17:00 NY
    h = ny.hour.values
    g = pd.DataFrame({'dia': dia, 'H': d.High.values, 'L': d.Low.values, 'h': h})
    tot = g.groupby('dia').agg(H=('H', 'max'), L=('L', 'min'))
    prev = tot.shift(1)
    asia = g[(g.h >= 20) & (g.h <= 23)].groupby('dia').agg(H=('H', 'max'), L=('L', 'min'))
    lon = g[(g.h >= 2) & (g.h < 5)].groupby('dia').agg(H=('H', 'max'), L=('L', 'min'))
    m = lambda t, c: pd.Series(dia).map(t[c]).values
    pH, pL, aH, aL, lH, lL = m(prev, 'H'), m(prev, 'L'), m(asia, 'H'), m(asia, 'L'), m(lon, 'H'), m(lon, 'L')
    out = []
    for k in range(len(d)):
        nv, hk = {}, h[k]
        if not math.isnan(pH[k]):
            nv['máximo de ayer'] = (pH[k], -1); nv['mínimo de ayer'] = (pL[k], 1)
        if hk < 17 and not math.isnan(aH[k]):                                # Asia cerrada (después de las 00:00 NY)
            nv['máximo de Asia'] = (aH[k], -1); nv['mínimo de Asia'] = (aL[k], 1)
        if 5 <= hk < 17 and not math.isnan(lH[k]):
            nv['máximo de Londres'] = (lH[k], -1); nv['mínimo de Londres'] = (lL[k], 1)
        out.append(nv)
    return out, dia, h


def _ventana(t, cual):
    if not cual:
        return 'todo el día'
    hh = t.tz_convert(NY); x = hh.hour + hh.minute / 60
    for nom, (a, b) in (KILLZONES if cual == 'kz' else SILVER_BULLET).items():
        if a <= x < b:
            return nom
    return None


def _pivotes(x, alto, N):
    out = np.zeros(len(x), bool)
    for i in range(N, len(x) - N):
        w = x[i - N:i + N + 1]
        out[i] = x[i] == (w.max() if alto else w.min())
    return out


def escanear(d, mins=5, P=None, tick=0.01):
    """Recorre las velas en orden (solo información pasada). Devuelve (setups, barridos): cada setup con su historia de
    eventos [(tiempo de cierre de vela, evento, precio)] (nueva = orden colocada, activada, t2, stop, cancelada, caducada,
    cierre) y cada barrido tal como se vio al producirse (para el aviso previo). Mismo resultado en backtest y en directo."""
    P = dict(BASE, **(P or {}))
    H, L, C = d.High.values, d.Low.values, d.Close.values
    idx, n, N = d.index, len(d), P['N']
    atr = _atr(d)
    niv, dia, hny = _niveles(d)
    ph, pl = _pivotes(H, True, N), _pivotes(L, False, N)
    h1 = d.Close.resample('1h').last().dropna()
    tend = np.sign(h1.ewm(span=20, adjust=False).mean() - h1.ewm(span=50, adjust=False).mean())
    tend.index = tend.index + pd.Timedelta(hours=1)                          # conocida al cerrar la vela de 1 h
    tend_v = tend.reindex(idx + pd.Timedelta(minutes=mins), method='ffill').values
    fin = lambda k: idx[k] + pd.Timedelta(minutes=mins)
    usados, cands, setups, barridos, op = set(), [], [], [], None
    dia_act, n_dia, seguidas = None, 0, 0
    for j in range(max(30, 2 * N + 3), n):
        if dia[j] != dia_act:
            dia_act, n_dia, seguidas = dia[j], 0, 0
        # 1) operación en curso (orden pendiente o abierta)
        if op is not None:
            s, lado = op, op['lado']
            if s['estado'] == 'pendiente':
                if (H[j] >= s['t2']) if lado > 0 else (L[j] <= s['t2']):
                    s['estado'] = 'cancelada'; s['ev'].append((fin(j), 'cancelada', None))
                elif (L[j] <= s['ent']) if lado > 0 else (H[j] >= s['ent']):
                    s['estado'] = 'activada'; s['ev'].append((fin(j), 'activada', s['ent'])); n_dia += 1
                    if (L[j] <= s['sl']) if lado > 0 else (H[j] >= s['sl']):
                        s['estado'] = 'stop'; s['ev'].append((fin(j), 'stop', s['sl']))
                elif j - s['j0'] >= P['W'] or P['cierre_ny'] <= hny[j] < 17:
                    s['estado'] = 'caducada'; s['ev'].append((fin(j), 'caducada', None))
            elif s['estado'] == 'activada':
                if (L[j] <= s['sl']) if lado > 0 else (H[j] >= s['sl']):
                    s['estado'] = 'stop'; s['ev'].append((fin(j), 'stop', s['sl']))
                elif (H[j] >= s['t2']) if lado > 0 else (L[j] <= s['t2']):
                    s['estado'] = 't2'; s['ev'].append((fin(j), 't2', s['t2']))
                elif P['cierre_ny'] <= hny[j] < 17:
                    s['estado'] = 'cierre'; s['ev'].append((fin(j), 'cierre', C[j])); s['salida'] = C[j]
                    s['r_cierre'] = (C[j] - s['ent']) * lado / s['R']
            if s['estado'] in ('stop', 't2', 'cierre'):
                r = -1.0 if s['estado'] == 'stop' else P['rr'] if s['estado'] == 't2' else (C[j] - s['ent']) * lado / s['R']
                seguidas = seguidas + 1 if r < 0 else 0
            if s['estado'] not in ('pendiente', 'activada'):
                op = None
        # 2) barridos esperando el MSS
        sigue = []
        for c in cands:
            lado = c['lado']
            if (C[j] < c['nivel']) if lado > 0 else (C[j] > c['nivel']):
                continue                                                     # aceptó al otro lado: ruptura, no barrido
            if (L[j] < c['ext']) if lado > 0 else (H[j] > c['ext']):
                c['ext'] = L[j] if lado > 0 else H[j]
            if (C[j] > c['ref']) if lado > 0 else (C[j] < c['ref']):        # MSS
                fvg = None
                for k in range(j, c['i'] + 1, -1):                           # el FVG más reciente del desplazamiento
                    lo, hi = (H[k - 2], L[k]) if lado > 0 else (H[k], L[k - 2])
                    if hi - lo >= P['minfvg'] * atr[j] and hi > lo:
                        fvg = (lo, hi); break
                if fvg is None or op is not None or n_dia >= P['max_dia'] or seguidas >= P['max_perdidas']:
                    continue
                vent = _ventana(fin(j), P['ventanas'])
                if vent is None or (P['sesgo'] and tend_v[j] * lado <= 0):
                    continue
                lo, hi = fvg
                ent = (lo + hi) / 2 if P['entrada'] == 'ce' else (hi if lado > 0 else lo)
                sl = c['ext'] - lado * P['buf']
                R = abs(ent - sl)
                if R <= 0 or R < P['minR'] or ((C[j] <= ent) if lado > 0 else (C[j] >= ent)):
                    continue
                s = dict(c, fvg_lo=lo, fvg_hi=hi, ent=ent, sl=sl, R=R, t1=ent + lado * R, t2=ent + lado * P['rr'] * R, j0=j,
                         t_mss=fin(j), ventana=vent, h1=int(tend_v[j]) if not math.isnan(tend_v[j]) else 0,
                         atr=atr[j], estado='pendiente', id=f"F{'C' if lado > 0 else 'V'}{idx[c['i']]:%Y%m%d%H%M}")
                s['ev'] = [(fin(j), 'nueva', ent)]
                setups.append(s); op = s
                continue
            if j - c['i'] < P['M']:
                sigue.append(c)
        cands = sigue
        # 3) barridos nuevos en esta vela
        for nom, (precio, lado) in niv[j].items():
            clave = (dia[j], nom)
            if clave in usados:
                continue
            perfora = (L[j] < precio) if lado > 0 else (H[j] > precio)
            if not perfora:
                continue
            usados.add(clave)
            if not ((C[j] > precio) if lado > 0 else (C[j] < precio)):
                continue                                                     # cerró al otro lado: ruptura
            piv = ph if lado > 0 else pl
            refs = [p for p in range(j - 1, max(j - 120, 0), -1) if piv[p] and p + N <= j]
            if not refs:
                continue
            ref = (H if lado > 0 else L)[refs[0]]
            if (ref <= C[j]) if lado > 0 else (ref >= C[j]):
                continue
            ext = L[j] if lado > 0 else H[j]
            c = dict(lado=lado, i=j, nivel=precio, nombre=nom, ext=ext, ref=ref, t=fin(j),
                     limite=fin(j) + pd.Timedelta(minutes=mins * P['M']), id=f"B{'C' if lado > 0 else 'V'}{idx[j]:%Y%m%d%H%M}")
            cands.append(c); barridos.append(dict(c))
    return setups, barridos


def resultado(s, coste=0.40):
    """R neto de un setup cerrado (None si no llegó a entrar o sigue abierto)."""
    if s['estado'] == 'stop':
        return (-s['R'] - coste) / s['R']
    if s['estado'] == 't2':
        return (abs(s['t2'] - s['ent']) - coste) / s['R']
    if s['estado'] == 'cierre':
        return ((s['salida'] - s['ent']) * s['lado'] - coste) / s['R']
    return None


# ---------------------------------------------------------------- mensajes (vigía)
def _f(x):
    return f"{x:.2f}"


def mensaje(s, evento, t, tfl, fondeo_txt='', nota=''):
    """Texto de Telegram para cada evento del modelo FVG, con el porqué de la operación."""
    compra = s['lado'] > 0
    lado = 'COMPRA' if compra else 'VENTA'
    hora = t.tz_convert('Europe/Madrid').strftime('%H:%M')
    tit = f"· XAUUSD · FVG {tfl}"
    if evento == 'barrido':
        limite = s['limite'].tz_convert('Europe/Madrid').strftime('%H:%M')
        return '\n'.join([
            f"👀 <b>BARRIDO DE LIQUIDEZ {tit}</b> · {hora}",
            f"El precio ha perforado el <b>{s['nombre']}</b> (<code>{_f(s['nivel'])}</code>) hasta <code>{_f(s['ext'])}</code> "
            f"y ha cerrado de nuevo {'por encima' if compra else 'por debajo'}: posible trampa a los {'vendedores' if compra else 'compradores'}.",
            f"Para una <b>{lado}</b> falta el <b>MSS</b>: cierre de {tfl} {'por encima' if compra else 'por debajo'} de "
            f"<code>{_f(s['ref'])}</code> (último {'máximo' if compra else 'mínimo'}) antes de las {limite}, con un FVG en el impulso.",
            f"Se anula si una vela cierra {'por debajo' if compra else 'por encima'} de <code>{_f(s['nivel'])}</code>. Aún no hay orden."])
    if evento == 'nueva':
        pd_ = 'descuento' if compra else 'premium'
        por_que = [f"✅ Barrido del {s['nombre']} ({_f(s['nivel'])}) y vuelta: liquidez recogida.",
                   f"✅ MSS: cierre {'sobre' if compra else 'bajo'} {_f(s['ref'])} con desplazamiento (FVG de {_f(s['fvg_hi'] - s['fvg_lo'])} $).",
                   f"✅ Entrada en el 50 % del FVG, en {pd_} del impulso; el stop va detrás del barrido."]
        contra = []
        if s['h1'] * s['lado'] < 0:
            contra.append('⚠️ Va contra la tendencia de 1 h (EMA 20 frente a EMA 50).')
        else:
            por_que.append('✅ A favor de la tendencia de 1 h.')
        por_que.append(f"✅ Dentro de la killzone «{s['ventana']}»." if s['ventana'] != 'todo el día' else '')
        if s['R'] < 2:
            contra.append(f"⚠️ Stop corto ({_f(s['R'])} $): el spread (~0,40 $) es el {0.40 / s['R']:.0%} del riesgo.")
        contra.append('⚠️ Si hay dato macro (IPC, empleo, Fed) en los próximos 30 min, mejor no entrar.')
        return '\n'.join([
            f"🆕 <b>{'🟢' if compra else '🔴'} {lado} FVG PREPARADA {tit}</b> · {hora}",
            'Modelo ICT: barrido de liquidez + MSS + FVG', *([nota] if nota else []),
            '━━━━━━━━━━━━━━━━━━',
            f"👉 Orden <b>{'buy limit' if compra else 'sell limit'}</b> en <code>{_f(s['ent'])}</code> "
            f"(FVG <code>{_f(s['fvg_lo'])}</code> – <code>{_f(s['fvg_hi'])}</code>)",
            f"⛔ SL: <code>{_f(s['sl'])}</code> (riesgo {_f(s['R'])} $, detrás del barrido)",
            f"🎯 TP 2R: <code>{_f(s['t2'])}</code>",
            *([fondeo_txt] if fondeo_txt else []),
            f"⏳ Cancelar si llega a <code>{_f(s['t2'])}</code> sin entrar o tras {BASE['W']} velas.",
            '━━━━━━━━━━━━━━━━━━', '<b>Por qué puede funcionar</b>', *[x for x in por_que if x],
            '<b>Por qué puede fallar</b>', *contra])
    textos = {
        'activada': f"✅ <b>{lado} FVG ACTIVADA {tit}</b> en <code>{_f(s['ent'])}</code> ({hora}). SL <code>{_f(s['sl'])}</code> · TP <code>{_f(s['t2'])}</code>",
        't2': f"🎯 <b>TP FVG {tit}</b> {lado.lower()}: tocado <code>{_f(s['t2'])}</code> (+2R) a las {hora}. Operación terminada.",
        'stop': f"⛔ <b>STOP FVG {tit}</b> {lado.lower()}: tocado <code>{_f(s['sl'])}</code> (−1R) a las {hora}. La idea queda invalidada.",
        'cancelada': f"❌ <b>CANCELADA FVG {tit}</b> {lado.lower()} {_f(s['ent'])}: llegó al objetivo sin entrar. Quita la orden.",
        'caducada': f"⌛ <b>CADUCADA FVG {tit}</b> {lado.lower()} {_f(s['ent'])}: no volvió al FVG a tiempo. Quita la orden.",
        'cierre': f"🔚 <b>CIERRE FVG {tit}</b> {lado.lower()}: fin de la sesión de Nueva York; cerrar a mercado ({hora}).",
    }
    return textos[evento]
