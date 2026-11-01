"""Graficos del monitor XAU Fractal DayTrader (estilo TradingView oscuro).

grafico_plan   -> briefing: diario (contexto) + horario con impulso y zona Fibonacci
grafico_senal  -> senal: 5 minutos con zona, entrada, SL y TP de cada posicion + MACD
grafico_cierre -> cierre: la sesion del dia en 5 minutos con las operaciones
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd, numpy as np
from zoneinfo import ZoneInfo
import motor as M

MADRID = ZoneInfo('Europe/Madrid')
FONDO, REJILLA, TEXTO = '#131722', '#2a2e39', '#d1d4dc'
VERDE, ROJO = '#26a69a', '#ef5350'
NARANJA, AZUL, MORADO, TEAL, GRIS = '#ff9800', '#2962ff', '#ab47bc', '#00bcd4', '#787b86'


def _ejes(ax):
    ax.set_facecolor(FONDO)
    ax.tick_params(colors=TEXTO, labelsize=8)
    for s in ax.spines.values():
        s.set_color(REJILLA)
    ax.grid(color=REJILLA, linewidth=0.5)
    ax.yaxis.tick_right()


def _velas(ax, df):
    x = np.arange(len(df))
    sube = df.Close.values >= df.Open.values
    col = np.where(sube, VERDE, ROJO)
    ax.vlines(x, df.Low.values, df.High.values, color=col, linewidth=0.8)
    for i, (o, c, k) in enumerate(zip(df.Open.values, df.Close.values, col)):
        ax.add_patch(Rectangle((i - 0.35, min(o, c)), 0.7, max(abs(c - o), 1e-6), color=k))
    ax.set_xlim(-1, len(df) + 8)
    return x


def _ticks_hora(ax, idx, fmt='%H:%M', n=8):
    pos = np.linspace(0, len(idx) - 1, min(n, len(idx))).astype(int)
    ax.set_xticks(pos)
    ax.set_xticklabels([idx[i].tz_convert(MADRID).strftime(fmt) for i in pos])


def _nivel(ax, y, color, texto, x0=0, estilo='-', ancho=1.2):
    ax.axhline(y, color=color, linestyle=estilo, linewidth=ancho, xmin=0)
    lo, hi = ax.get_ylim(); hueco = (hi - lo) * 0.045
    usadas = ax.__dict__.setdefault('_etiquetas', [])
    yl = y
    while any(abs(yl - u) < hueco for u in usadas):
        yl += hueco if yl >= y else -hueco
        if abs(yl - y) > hueco * 6:
            yl = y - hueco
    usadas.append(yl)
    ax.annotate(f'{texto} {y:.2f}', xy=(1.0, yl), xycoords=('axes fraction', 'data'), xytext=(4, 0),
                textcoords='offset points', color='white', fontsize=8, va='center',
                bbox=dict(boxstyle='round,pad=0.25', fc=color, ec='none'))


def _titulo(fig, txt, sub):
    fig.text(0.01, 0.975, txt, color='white', fontsize=13, fontweight='bold', va='top')
    fig.text(0.01, 0.935, sub, color=TEXTO, fontsize=9, va='top')


def _zona(ax, e):
    if e.get('pauta'):
        ax.axhspan(e['zona_a'], e['zona_b'], color=TEAL, alpha=0.15)
        ax.axhline(e['fibo50'], color=TEAL, linestyle=':', linewidth=0.9)
        ax.axhline(e['ext'], color=GRIS, linewidth=1.2)
        ax.axhline(e['ori'], color=GRIS, linewidth=1.2)


def _dir(d):
    return {1: 'ALCISTA', -1: 'BAJISTA', 0: 'NEUTRAL'}[int(d)]


def grafico_plan(win, e, ruta, P=M.V23):
    m5, h1, d1 = M.prep(win)
    h = h1[h1.index <= e['t']].tail(96)
    d = d1.tail(70)
    fig = plt.figure(figsize=(11, 7.2), facecolor=FONDO)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.3], hspace=0.3, top=0.86, bottom=0.05, left=0.02, right=0.88)
    ax = fig.add_subplot(gs[0]); _ejes(ax)
    x = _velas(ax, h)
    ax.plot(x, M.ema(h1.Close, P['h1f']).reindex(h.index).values, color=NARANJA, lw=1.2, label='EMA 20 H1')
    ax.plot(x, M.ema(h1.Close, P['h1s']).reindex(h.index).values, color=AZUL, lw=1.2, label='EMA 50 H1')
    _zona(ax, e)
    if e.get('pauta'):
        _nivel(ax, e['zona_b'], TEAL, '2/3' if e['bias'] < 0 else '1/3')
        _nivel(ax, e['zona_a'], TEAL, '1/3' if e['bias'] < 0 else '2/3')
        _nivel(ax, e['ext'], GRIS, 'Extremo')
        _nivel(ax, e['ori'], GRIS, 'Origen')
    ax.axhline(e['pdh'], color=MORADO, linestyle='--', lw=1); ax.axhline(e['pdl'], color=MORADO, linestyle='--', lw=1)
    _nivel(ax, e['pdh'], MORADO, 'Máx ayer', estilo='--'); _nivel(ax, e['pdl'], MORADO, 'Mín ayer', estilo='--')
    _nivel(ax, e['precio'], AZUL, 'Precio')
    _ticks_hora(ax, h.index, '%d/%m %Hh')
    ax.legend(loc='upper left', fontsize=8, facecolor=FONDO, edgecolor=REJILLA, labelcolor=TEXTO)
    ax.set_title('2) HORARIO (H1): tendencia, impulso y zona Fibonacci 1/3–2/3', color=TEXTO, fontsize=10, loc='left')
    ax2 = fig.add_subplot(gs[1]); _ejes(ax2)
    xd = _velas(ax2, d)
    ax2.plot(xd, M.ema(d1.Close, P['d1f']).reindex(d.index).values, color=NARANJA, lw=1)
    ax2.plot(xd, M.ema(d1.Close, P['d1s']).reindex(d.index).values, color=AZUL, lw=1)
    _ticks_hora(ax2, d.index, '%d/%m', 7)
    ax2.set_title(f"1) DIARIO: {_dir(e['bias'])}  (EMA 20 / 50)", color=TEXTO, fontsize=10, loc='left')
    sub = (f"Diario {_dir(e['bias'])} · Horario {_dir(e['h1'])} · ADX H1 {e['adx']:.1f}"
           f"{' (OK)' if e['adx_ok'] else ' (sin fuerza)'} · {pd.Timestamp(e['t']).tz_convert(MADRID):%d/%m %H:%M} (España)")
    _titulo(fig, 'XAUUSD · Plan del día', sub)
    fig.savefig(ruta, dpi=110, facecolor=FONDO); plt.close(fig)
    return ruta


def _m5_con_indicadores(win, hasta, barras, P):
    m5, _, _ = M.prep(win)
    m5 = m5[m5.index <= hasta]
    f = M.ema(m5.Close, P['m5f']); s = M.ema(m5.Close, P['m5s']); _, _, mh = M.macd(m5.Close)
    v = m5.tail(barras)
    return v, f.reindex(v.index).values, s.reindex(v.index).values, mh.reindex(v.index).values


def _panel_m5(fig, v, f, s, mh, titulo_sup):
    gs = fig.add_gridspec(2, 1, height_ratios=[3.4, 1], hspace=0.12, top=0.86, bottom=0.06, left=0.02, right=0.86)
    ax = fig.add_subplot(gs[0]); _ejes(ax)
    x = _velas(ax, v)
    ax.plot(x, f, color=NARANJA, lw=1.2, label='EMA 9'); ax.plot(x, s, color=AZUL, lw=1.2, label='EMA 21')
    ax.legend(loc='upper left', fontsize=8, facecolor=FONDO, edgecolor=REJILLA, labelcolor=TEXTO)
    ax.set_title(titulo_sup, color=TEXTO, fontsize=10, loc='left'); ax.set_xticklabels([])
    axm = fig.add_subplot(gs[1], sharex=ax); _ejes(axm)
    axm.bar(x, mh, color=np.where(mh >= 0, VERDE, ROJO), width=0.7)
    axm.axhline(0, color=GRIS, lw=0.8)
    axm.set_title('MACD (12,26,9) histograma', color=TEXTO, fontsize=8, loc='left', pad=2)
    _ticks_hora(axm, v.index)
    plt.setp(ax.get_xticklabels(), visible=False)
    return ax, axm, x


def grafico_senal(win, e, s, ruta, P=M.V23):
    t = s['t']
    v, f, sl, mh = _m5_con_indicadores(win, t, 84, P)
    fig = plt.figure(figsize=(11, 7), facecolor=FONDO)
    ax, _, x = _panel_m5(fig, v, f, sl, mh, '3) 5 MINUTOS: cruce EMA 9/21 + MACD a favor')
    _zona(ax, e)
    lado = 'VENTA' if s['dir'] < 0 else 'COMPRA'
    ax.annotate('▼ VENTA' if s['dir'] < 0 else '▲ COMPRA', xy=(x[-1], v.High.iloc[-1] if s['dir'] < 0 else v.Low.iloc[-1]),
                xytext=(0, 14 if s['dir'] < 0 else -18), textcoords='offset points', ha='center',
                color=ROJO if s['dir'] < 0 else VERDE, fontsize=10, fontweight='bold')
    _nivel(ax, s['entry'], AZUL, 'Entrada')
    _nivel(ax, s['slA'], ROJO, 'SL A'); _nivel(ax, s['tpA'], VERDE, 'TP A')
    if s['useB']:
        _nivel(ax, s['slB'], '#b71c1c', 'SL B', estilo='--'); _nivel(ax, s['tpB'], '#1b5e20', 'TP B', estilo='--')
    lo = min(v.Low.min(), s['slA'], s['tpA']); hi = max(v.High.max(), s['slA'], s['tpA'])
    if s['useB']:
        lo, hi = min(lo, s['slB'], s['tpB']), max(hi, s['slB'], s['tpB'])
    ax.set_ylim(lo - (hi - lo) * 0.05, hi + (hi - lo) * 0.05)
    _titulo(fig, f'XAUUSD · SEÑAL DE {lado}',
            f"{pd.Timestamp(t + pd.Timedelta(minutes=5)).tz_convert(MADRID):%d/%m %H:%M} (España) · retroceso {s['retr']*100:.0f}% · "
            f"Diario {_dir(e['bias'])} · ADX H1 {e['adx']:.1f}")
    fig.savefig(ruta, dpi=110, facecolor=FONDO); plt.close(fig)
    return ruta


def grafico_cierre(win, e, ops, dia, ruta, P=M.V23):
    ini = dia + pd.Timedelta(hours=6); fin = dia + pd.Timedelta(hours=P['close'], minutes=5)
    m5, _, _ = M.prep(win)
    m5 = m5[(m5.index >= ini - pd.Timedelta(hours=2)) & (m5.index < fin)]
    f = M.ema(m5.Close, P['m5f']); s = M.ema(m5.Close, P['m5s']); _, _, mh = M.macd(m5.Close)
    v = m5[m5.index >= ini]
    fig = plt.figure(figsize=(11, 7), facecolor=FONDO)
    ax, _, x = _panel_m5(fig, v, f.reindex(v.index).values, s.reindex(v.index).values, mh.reindex(v.index).values,
                         'Sesión del día en 5 minutos (zona sombreada = horario de entradas 12:00–21:00)')
    dentro = (v.index >= dia + pd.Timedelta(hours=P['start'])) & (v.index < dia + pd.Timedelta(hours=P['end']))
    if dentro.any():
        i0, i1 = np.argmax(dentro), len(dentro) - np.argmax(dentro[::-1]) - 1
        ax.axvspan(i0, i1, color='#2962ff', alpha=0.06)
    for op in ops:
        i = v.index.get_indexer([op['t']], method='nearest')[0]
        j = v.index.get_indexer([op.get('fin', v.index[-1])], method='nearest')[0]
        col = VERDE if op['dir'] > 0 else ROJO
        ax.scatter([i], [op['entry']], marker='^' if op['dir'] > 0 else 'v', s=160, color=col, edgecolors='white', zorder=5)
        ax.hlines(op['entry'], i, j, colors=AZUL, linewidth=1.2)
        ax.hlines(op['slA'], i, j, colors=ROJO, linewidth=1.2)
        ax.hlines(op['tpA'], i, j, colors=VERDE, linewidth=1.2)
        if op.get('B') is not None and op.get('usoB'):
            ax.hlines([op['slB'], op['tpB']], i, j, colors=['#b71c1c', '#1b5e20'], linestyles='--', linewidth=1)
        ax.annotate(f"{'COMPRA' if op['dir'] > 0 else 'VENTA'} {op.get('res', 0):+.0f} $", xy=(i, op['entry']),
                    xytext=(8, 14 if op['dir'] < 0 else -18), textcoords='offset points', color='white', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.25', fc=col, ec='none'))
    tot = sum(o.get('res', 0) for o in ops)
    _titulo(fig, f"XAUUSD · Cierre del día {dia.tz_convert(MADRID):%d/%m/%Y}",
            f"Operaciones: {len(ops)} · Resultado (0,02 + 0,01 lotes): {tot:+.0f} $ · Cierre {v.Close.iloc[-1]:.2f}"
            if len(v) else 'Sin datos de sesión')
    fig.savefig(ruta, dpi=110, facecolor=FONDO); plt.close(fig)
    return ruta
