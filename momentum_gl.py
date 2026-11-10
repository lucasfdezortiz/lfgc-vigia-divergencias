"""Momentum de dos velas de Gian Luca Bruni (OPERATIVA §27), para el US100 en 5 min.

- Vela válida: cuerpo mayor que la mecha más larga. Tres válidas seguidas del mismo color forman una zona; la última es la
  estructura «actual» y la anterior, la «periférica». Corto si el precio ha cerrado por debajo de las dos zonas verdes
  (demanda); largo si ha cerrado por encima de las dos rojas (oferta).
- Gatillo: dos velas seguidas en la dirección, cada una con cuerpo >= k × ATR(20).
- Entrada a mercado al cierre de la segunda vela; SL en el extremo de la primera; TP fijo a 1,5R. Una operación a la vez,
  entradas de 09:30 a 15:45 de Nueva York y cierre forzoso a las 16:00.
Backtest (backtest/gianluca_momentum.py, que usa este mismo contexto): US100 5 min 2023-26, 562 ops, E +0,09R neta,
PF 1,20 / 1,17. Lo usan el vigía (avisos), el punto de control y el backtest.
"""
import numpy as np, pandas as pd

BASE = dict(k=1.0, rr=1.5, tz='America/New_York', ini=9.5, fin=15.75, cierre=16.0, contexto='alineadas')
MADRID = 'Europe/Madrid'


def zonas_contexto(d):
    """Por vela (al cierre): contexto +1 largo / -1 corto con la estructura actual y con actual + periférica alineadas,
    si la vela es válida, y las zonas vigentes al final ([lo, hi] de las dos últimas verdes y rojas)."""
    O, H, L, C = (d[k].values for k in ('Open', 'High', 'Low', 'Close'))
    cuerpo = C - O
    mecha = np.maximum(H - np.maximum(O, C), np.minimum(O, C) - L)
    valida = np.abs(cuerpo) > mecha
    color = np.where(valida & (cuerpo > 0), 1, np.where(valida & (cuerpo < 0), -1, 0))
    n = len(d)
    act, ali = np.zeros(n, int), np.zeros(n, int)
    dem, ofe = [], []                            # zonas verdes (demanda) y rojas (oferta): [lo, hi]
    run_c, run_n, run_lo, run_hi = 0, 0, 0.0, 0.0
    for i in range(n):
        c = color[i]
        if c != 0 and c == run_c:
            run_n += 1; run_lo = min(run_lo, L[i]); run_hi = max(run_hi, H[i])
        else:
            run_c, run_n, run_lo, run_hi = c, (1 if c else 0), L[i], H[i]
        if run_n >= 3:                           # tres válidas seguidas: zona (se amplía mientras siga la racha)
            lst = dem if run_c > 0 else ofe
            if run_n == 3:
                lst.append([run_lo, run_hi])
                del lst[:-2]
            else:
                lst[-1] = [run_lo, run_hi]
        corto = len(dem) >= 1 and C[i] < dem[-1][0]
        largo = len(ofe) >= 1 and C[i] > ofe[-1][1]
        act[i] = -1 if corto and not largo else (1 if largo and not corto else 0)
        corto2 = corto and len(dem) == 2 and C[i] < dem[0][0]
        largo2 = largo and len(ofe) == 2 and C[i] > ofe[0][1]
        ali[i] = -1 if corto2 and not largo2 else (1 if largo2 and not corto2 else 0)
    return act, ali, valida, dict(dem=[list(z) for z in dem], ofe=[list(z) for z in ofe])


def _atr(d):
    H, L, C = d.High.values, d.Low.values, d.Close.values
    tr = np.maximum(H - L, np.maximum(abs(H - np.roll(C, 1)), abs(L - np.roll(C, 1))))
    return pd.Series(tr).rolling(20).mean().values


def escanear(d, mins=5, P=None):
    """Operaciones sobre velas cerradas. Cada una: id, lado, ent, sl, t1 = t2 = TP, R, umbral, ev [(t, evento, precio)]
    con 'activada' (entrada) y luego 't2', 'stop' o 'cierre' (si en una misma vela se tocan SL y TP, cuenta el SL)."""
    P = dict(BASE, **(P or {}))
    O, H, L, C = (d[k].values for k in ('Open', 'High', 'Low', 'Close'))
    act, ali, _, _ = zonas_contexto(d)
    ctx = ali if P['contexto'] == 'alineadas' else act
    atr = _atr(d)
    loc = d.index.tz_convert(P['tz']); hora = loc.hour + loc.minute / 60
    paso = pd.Timedelta(minutes=mins)
    ops, op = [], None
    for i in range(22, len(d)):
        if op is not None:                       # gestión de la operación abierta
            lado = op['lado']
            if hora[i] >= P['cierre'] or loc[i].date() != op['dia']:
                op['r_cierre'] = lado * (O[i] - op['ent']) / op['R']
                op['ev'].append((d.index[i], 'cierre', O[i])); op = None
            elif (L[i] <= op['sl']) if lado > 0 else (H[i] >= op['sl']):
                op['ev'].append((d.index[i] + paso, 'stop', op['sl'])); op = None
            elif (H[i] >= op['t2']) if lado > 0 else (L[i] <= op['t2']):
                op['ev'].append((d.index[i] + paso, 't2', op['t2'])); op = None
            if op is not None:
                continue
        if not (P['ini'] <= hora[i] < P['fin']):
            continue
        for lado in (1, -1):
            umbral = P['k'] * atr[i - 2]
            if not (lado * (C[i] - O[i]) >= umbral and lado * (C[i - 1] - O[i - 1]) >= umbral and umbral > 0):
                continue
            if ctx[i] != lado:
                continue
            ent = C[i]; sl = L[i - 1] if lado > 0 else H[i - 1]; R = lado * (ent - sl)
            if R <= 0:
                continue
            tp = ent + lado * P['rr'] * R
            t = d.index[i] + paso
            op = dict(id=f"G{'C' if lado > 0 else 'V'}{t:%Y%m%d%H%M}", lado=lado, ent=ent, sl=sl, t1=tp, t2=tp, R=R,
                      umbral=umbral, dia=loc[i].date(), etq=' (momentum GL)', ev=[(t, 'activada', ent)])
            ops.append(op)
            break
    return ops


def estado(d, mins=5, P=None):
    """Foto para el punto de control: contexto, zonas, umbral de las velas, operación abierta y ops de hoy."""
    P = dict(BASE, **(P or {}))
    act, ali, _, z = zonas_contexto(d)
    ops = escanear(d, mins, P)
    hoy = d.index[-1].tz_convert(P['tz']).date()
    abierta = next((s for s in ops if s['ev'][-1][1] == 'activada'), None)
    loc = d.index[-1].tz_convert(P['tz']); h = loc.hour + loc.minute / 60
    return dict(act=int(act[-1]), ali=int(ali[-1]), zonas=z, umbral=P['k'] * _atr(d)[-1], op=abierta,
                n_hoy=sum(s['dia'] == hoy for s in ops), precio=float(d.Close.iloc[-1]),
                en_sesion=P['ini'] <= h < P['fin'] and loc.weekday() < 5)


def _hora(h, P=BASE):
    """Hora de Nueva York (en horas decimales) pasada a hora de España, para hoy."""
    t = pd.Timestamp.now(tz=P['tz']).normalize() + pd.Timedelta(hours=h)
    return t.tz_convert(MADRID).strftime('%H:%M')


def texto_control(sim, e, ud='pt'):
    f = lambda x: f"{x:.2f}"
    ctx = {1: '🟢 largo (actual y periférica rotas al alza)', -1: '🔴 corto (actual y periférica rotas a la baja)'}.get(
        e['ali'], {1: '🟡 solo la actual es alcista', -1: '🟡 solo la actual es bajista'}.get(e['act'], '⚪ sin dirección'))
    l = [f"👉 <b>{sim}</b> ({f(e['precio'])}): contexto {ctx}"]
    z = e['zonas']
    if z['dem'] or z['ofe']:
        l.append('   🧱 Zonas: ' + ' · '.join(
            [f"demanda {f(a)}–{f(b)}" for a, b in reversed(z['dem'])] + [f"oferta {f(a)}–{f(b)}" for a, b in reversed(z['ofe'])]))
    if e['op']:
        s = e['op']
        l.append(f"   📌 Operación abierta: {'compra' if s['lado'] > 0 else 'venta'} en {f(s['ent'])} · SL {f(s['sl'])} · TP {f(s['t2'])}")
    elif e['ali']:
        l.append(f"   ⏳ Falta el gatillo: dos velas de 5 min {'alcistas' if e['ali'] > 0 else 'bajistas'} seguidas con cuerpo ≥ "
                 f"{e['umbral']:.1f} {ud} cada una, entre las {_hora(BASE['ini'])} y las {_hora(BASE['fin'])} (España).")
    else:
        l.append('   ⏳ Falta que el precio rompa las dos zonas del mismo lado (actual y periférica).')
    l.append(f"   Hoy: {e['n_hoy']} operación{'es' if e['n_hoy'] != 1 else ''}.")
    return l


def mensaje(s, evento, t, tfl='5 min', fondeo_txt='', nota='', sim='US100', coste=1.5, ud='pt'):
    lado = 'COMPRA' if s['lado'] > 0 else 'VENTA'
    hora = t.tz_convert(MADRID).strftime('%H:%M')
    f = lambda x: f"<code>{x:.2f}</code>"
    tit = f"{sim} · Momentum GL · {tfl}"
    if evento == 'activada':
        return '\n'.join([
            f"🆕 {'🟢' if s['lado'] > 0 else '🔴'} <b>{lado} MOMENTUM GL · {sim} · {tfl}</b> · {hora}",
            f"Estructura actual y periférica {'alcistas' if s['lado'] > 0 else 'bajistas'} + dos velas de {tfl} seguidas con "
            f"cuerpo ≥ {s['umbral']:.1f} {ud} (1 ATR)",
            *([nota] if nota else []),
            '━━━━━━━━━━━━━━━━━━',
            '📋 <b>ORDEN PARA MT5</b>',
            f"Tipo: <b>{'buy' if s['lado'] > 0 else 'sell'} a mercado</b> (ya: la entrada es el cierre de la 2.ª vela)",
            f"Entrada: {f(s['ent'])}",
            f"SL: {f(s['sl'])} ({s['R']:.1f} {ud}, extremo de la 1.ª vela)",
            f"TP: {f(s['t2'])} ({BASE['rr']}R)",
            *([fondeo_txt] if fondeo_txt else []),
            '━━━━━━━━━━━━━━━━━━',
            f"⚠️ Si en MT5 el precio ya se ha alejado de la entrada, recalcula los lotes con la distancia real al SL; "
            f"si ya ha recorrido 0,5R hacia el TP, no entres.",
            f"⏳ Si no toca SL ni TP, se cierra a mercado a las {_hora(BASE['cierre'])} (16:00 de Nueva York)."])
    if evento == 't2':
        return f"🎯 <b>TP · {tit}</b> {lado.lower()}: tocado {f(s['t2'])} (+{BASE['rr']}R) a las {hora}. Operación terminada."
    if evento == 'stop':
        return f"⛔ <b>STOP · {tit}</b> {lado.lower()}: tocado {f(s['sl'])} (−1R) a las {hora}."
    if evento == 'cierre':
        return (f"🔚 <b>CIERRE · {tit}</b> {lado.lower()}: fin de la sesión de Nueva York; cierra a mercado "
                f"({s.get('r_cierre', 0):+.1f}R aprox.) a las {hora}.")
    return ''
