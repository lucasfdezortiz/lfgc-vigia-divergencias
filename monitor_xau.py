#!/usr/bin/env python3
"""Monitor XAU Fractal DayTrader v2.3 -> alertas por Telegram (con graficos).

Aplica exactamente las reglas del backtest (motor.py) sobre el precio del oro en
tiempo real (consulta cada segundo) y envia por Telegram:
  - Plan del dia (08:00 hora de Espana) con grafico diario + horario.
  - PREPARADO: el precio esta en la zona 1/3-2/3 a favor de la tendencia.
  - SENAL: cruce EMA 9/21 + MACD en 5 min -> posiciones, lotes, SL y TP + grafico.
  - Gestion al segundo: TP A (runner a break-even), stops, objetivo runner, cierre.
  - Cierre del dia (20:05 UTC = 22:05 Espana) con resultado y grafico de la sesion.

Uso:
  python3 monitor_xau.py --setup                 configura el bot de Telegram (una vez)
  python3 monitor_xau.py --prueba                envia el plan del dia actual y sale
  python3 monitor_xau.py --simular [AAAA-MM-DD]  reproduce un dia pasado por Telegram
  python3 monitor_xau.py                         arranca el monitor

Precios: oro al contado de Swissquote (tiempo real, publico) e historico de
Dukascopy. Pueden diferir unos centimos de tu broker: confirma en MT5.
"""
import math, json, os, sys, time, lzma, uuid, html, urllib.request, urllib.parse, datetime as dt, traceback, subprocess
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import motor as M
import graficos as G

AQUI = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('XAU_DATA') or os.path.join(AQUI, 'data', 'xau_m1.pkl')
IMG = os.path.join(AQUI, 'data')
CFG = os.path.join(AQUI, 'config_telegram.json')
LOG = os.path.join(AQUI, 'monitor.log')
UA = {'User-Agent': 'Mozilla/5.0'}
SQ = 'https://forex-data-feed.swissquote.com/public-quotes/bboquotes/instrument/XAU/USD'
MADRID = ZoneInfo('Europe/Madrid')
P = dict(M.V23)
if os.environ.get('XAU_CUENTA'):
    P['balance'] = float(os.environ['XAU_CUENTA'])
LOTE_A, LOTE_B = P['lotA'], P['lotB']
# Mismas reglas sin restriccion horaria (02:00-22:00 Espana): solo para avisos 'FUERA DE HORARIO'
P_LIBRE = dict(P, start=0, end=20)
CUENTA = P['balance']
SEP = '━━━━━━━━━━━━━━━━━━'
HORA_PLAN = 8   # plan del dia a las 08:00 (hora de Espana)
# Cuenta de fondeo (FTMO 10K): lotes para arriesgar FONDEO_RIESGO repartido A:B = 2:1
FONDEO = os.environ.get('XAU_FONDEO_NOMBRE', 'FTMO 10K')
FONDEO_CAP = float(os.environ.get('XAU_FONDEO_CAPITAL', 10000))
FONDEO_RIESGO = float(os.environ.get('XAU_FONDEO_RIESGO', 0.015))
FONDEO_EXTRA = float(os.environ.get('XAU_FONDEO_RIESGO_EXTRA', FONDEO_RIESGO))   # señales fuera de horario (no validadas)
PREFIJO = ''


def log(msg):
    line = f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%S} UTC | {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


DIAS = {'Mon': 'lun', 'Tue': 'mar', 'Wed': 'mié', 'Thu': 'jue', 'Fri': 'vie', 'Sat': 'sáb', 'Sun': 'dom'}


def hora_es(ts, fmt='%H:%M'):
    txt = pd.Timestamp(ts).tz_convert(MADRID).strftime(fmt)
    for en, es in DIAS.items():
        txt = txt.replace(en, es)
    return txt


def px(v):
    return f'<code>{v:.2f}</code>'


def usd(lotes, dist):
    return lotes * 100 * dist


def lotes_fondeo(s, riesgo=None):
    """Lotes A y B en la cuenta de fondeo: riesgo total = riesgo (FONDEO_RIESGO) del capital, A el doble de lotes que B."""
    obj = FONDEO_CAP * (riesgo or FONDEO_RIESGO)
    dA = abs(s['entry'] - s['slA']); dB = abs(s['entry'] - s['slB'])
    if s.get('useB', s.get('usoB')):
        k = obj / (100 * (2 * dA + dB))
        return max(0.01, math.floor(2 * k * 100) / 100), max(0.01, math.floor(k * 100) / 100)
    return max(0.01, math.floor(obj / (100 * dA) * 100) / 100), 0.0


# ---------------------------------------------------------------- Telegram
def cfg():
    if os.environ.get('TELEGRAM_TOKEN'):                     # en la nube: secretos de GitHub
        return {'token': os.environ['TELEGRAM_TOKEN'], 'chat_id': os.environ['TELEGRAM_CHAT']}
    return json.load(open(CFG)) if os.path.exists(CFG) else None


ES_MAC = sys.platform == 'darwin'


def aviso_mac(texto):
    if os.environ.get('XAU_SIN_AVISO_MAC') or not ES_MAC:
        return
    try:
        limpio = texto[:180].replace('\\', ' ').replace('"', "'")
        subprocess.run(['osascript', '-e', f'display notification "{limpio}" with title "XAU Fractal"'],
                       timeout=5, capture_output=True)
    except Exception:
        pass


def _api(metodo, campos, foto=None):
    c = cfg()
    url = f"https://api.telegram.org/bot{c['token']}/{metodo}"
    if foto is None:
        req = urllib.request.Request(url, data=urllib.parse.urlencode(campos).encode())
    else:
        b = uuid.uuid4().hex
        cuerpo = b''
        for k, v in campos.items():
            cuerpo += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        cuerpo += (f'--{b}\r\nContent-Disposition: form-data; name="photo"; filename="grafico.png"\r\n'
                   f'Content-Type: image/png\r\n\r\n').encode() + open(foto, 'rb').read() + f'\r\n--{b}--\r\n'.encode()
        req = urllib.request.Request(url, data=cuerpo, headers={'Content-Type': f'multipart/form-data; boundary={b}'})
    urllib.request.urlopen(req, timeout=30).read()


def tg(texto, foto=None, pie=None):
    """Envia texto HTML; si hay grafico, se envia despues con un pie corto."""
    texto = (f'🧪 <i>{PREFIJO}</i>\n' if PREFIJO else '') + texto
    plano = html.unescape(__import__('re').sub('<[^>]+>', '', texto))
    log('ENVIO: ' + plano.replace('\n', ' | ')[:300])
    aviso_mac(plano.split('\n')[1 if PREFIJO else 0])
    if not cfg():
        log('Telegram sin configurar (ejecuta --setup)')
        return
    try:
        _api('sendMessage', {'chat_id': cfg()['chat_id'], 'text': texto, 'parse_mode': 'HTML'})
        if foto:
            _api('sendPhoto', {'chat_id': cfg()['chat_id'], 'caption': pie or '', 'parse_mode': 'HTML'}, foto)
    except Exception as e:
        log(f'Error enviando a Telegram: {e}')
    copia_github(plano)


def copia_github(plano):
    """En la nube: copia el aviso como comentario de la issue AVISOS_ISSUE (la lee Claude)."""
    n, repo, token = (os.environ.get(k) for k in ('AVISOS_ISSUE', 'GITHUB_REPOSITORY', 'GITHUB_TOKEN'))
    if not (n and repo and token):
        return
    try:
        cuerpo = json.dumps({'body': f'`{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC`\n\n{plano}'}).encode()
        req = urllib.request.Request(f'https://api.github.com/repos/{repo}/issues/{n}/comments', data=cuerpo,
                                     headers={'Authorization': f'Bearer {token}',
                                              'Accept': 'application/vnd.github+json'})
        urllib.request.urlopen(req, timeout=15).read()
    except Exception as e:
        log(f'Error copiando el aviso a GitHub: {e}')


def setup():
    print(__doc__)
    print('1) En Telegram abre @BotFather, escribe /newbot y sigue los pasos.')
    print('2) Copia el token que te da BotFather (no lo compartas con nadie).')
    token = input('\nPega aqui el token y pulsa Enter: ').strip()
    base = f'https://api.telegram.org/bot{token}'
    try:
        me = json.loads(urllib.request.urlopen(f'{base}/getMe', timeout=20).read())['result']
    except Exception:
        print('\nEse token no es valido. Copialo de nuevo de @BotFather (/mybots -> tu bot -> API Token).')
        return
    urllib.request.urlopen(f'{base}/deleteWebhook', timeout=20).read()
    print(f"\nToken correcto: bot @{me['username']}.")
    print(f"3) Abre https://t.me/{me['username']} en Telegram, pulsa \"Iniciar\" y escribele \"hola\".")
    print('   Esperando tu mensaje (hasta 3 minutos)...')
    chats = []
    for _ in range(36):
        r = json.loads(urllib.request.urlopen(f'{base}/getUpdates', timeout=20).read())
        chats = [u[k]['chat']['id'] for u in r.get('result', [])
                 for k in ('message', 'edited_message', 'my_chat_member') if k in u]
        if chats:
            break
        time.sleep(5)
    if not chats:
        print('No ha llegado ningun mensaje. Vuelve a abrir el configurador.')
        return
    json.dump({'token': token, 'chat_id': chats[-1]}, open(CFG, 'w'))
    os.chmod(CFG, 0o600)
    tg('✅ <b>Monitor XAU Fractal DayTrader conectado.</b>\nAquí recibirás las alertas de XAUUSD.')
    print('\nListo. Revisa Telegram.')


# ---------------------------------------------------------------- Datos
def velas_hora(h):
    u = f"https://datafeed.dukascopy.com/datafeed/XAUUSD/{h.year}/{h.month-1:02d}/{h.day:02d}/{h.hour:02d}h_ticks.bi5"
    r = None
    for k in range(3):
        try:
            r = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=90).read()
            break
        except Exception:
            time.sleep(3 + 5 * k)
    if not r:
        return None
    raw = lzma.decompress(r)
    n = len(raw) // 20
    if n == 0:
        return None
    a = np.frombuffer(raw[:n * 20], dtype=[('ms', '>i4'), ('ask', '>i4'), ('bid', '>i4'), ('av', '>f4'), ('bv', '>f4')])
    s = pd.Series(a['bid'] / 1000.0, index=h + pd.to_timedelta(a['ms'].astype('int64'), unit='ms'))
    o = s.resample('1min').ohlc().dropna()
    o.columns = ['Open', 'High', 'Low', 'Close']
    o['Vol'] = s.resample('1min').count().reindex(o.index).astype(float)
    return o


def completar_historico(m1, horas_atras=None):
    ahora = pd.Timestamp.now(tz='UTC').floor('1h')
    desde = (ahora - pd.Timedelta(hours=horas_atras)) if horas_atras else m1.index.max().floor('1h')
    horas = [h for h in pd.date_range(desde, ahora - pd.Timedelta(hours=1), freq='1h')
             if h.weekday() != 5 and m1.loc[h:h + pd.Timedelta(minutes=59)].shape[0] < 50]
    if not horas:
        return m1
    log(f'Completando historico: {len(horas)} horas...')
    with ThreadPoolExecutor(6) as ex:
        partes = [p for p in ex.map(velas_hora, horas) if p is not None and len(p)]
    if partes:
        m1 = pd.concat([m1] + partes)
        m1 = m1[~m1.index.duplicated(keep='last')].sort_index()
    return m1


def vela_dia(day):
    """Velas M1 de un dia completo (un solo fichero de Dukascopy): para crear el historico desde cero."""
    u = f"https://datafeed.dukascopy.com/datafeed/XAUUSD/{day.year}/{day.month-1:02d}/{day.day:02d}/BID_candles_min_1.bi5"
    for k in range(5):
        try:
            raw = lzma.decompress(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60).read())
            break
        except Exception:
            time.sleep(3 + 4 * k)
    else:
        return None
    a = np.frombuffer(raw[:len(raw) // 24 * 24], dtype=[('t', '>i4'), ('o', '>i4'), ('c', '>i4'), ('l', '>i4'), ('h', '>i4'), ('v', '>f4')])
    if not len(a):
        return None
    idx = pd.Timestamp(day, tz='UTC') + pd.to_timedelta(a['t'].astype('int64'), unit='s')
    d = pd.DataFrame({'Open': a['o'] / 1000, 'High': a['h'] / 1000, 'Low': a['l'] / 1000, 'Close': a['c'] / 1000,
                      'Vol': a['v'].astype(float)}, index=idx)
    return d[d.Vol > 0]


def historico_inicial(dias=210):
    hoy = pd.Timestamp.now(tz='UTC').normalize()
    dias_l = [(hoy - pd.Timedelta(days=k)).date() for k in range(dias, 0, -1)]
    dias_l = [x for x in dias_l if x.weekday() != 5]
    log(f'Creando historico inicial: {len(dias_l)} dias...')
    with ThreadPoolExecutor(8) as ex:
        partes = [p for p in ex.map(vela_dia, dias_l) if p is not None and len(p)]
    m1 = pd.concat(partes).sort_index()
    return completar_historico(m1[~m1.index.duplicated()], horas_atras=int((pd.Timestamp.now(tz='UTC') - hoy).total_seconds() // 3600) + 1)


def cotizacion():
    r = json.loads(urllib.request.urlopen(urllib.request.Request(SQ, headers=UA), timeout=10).read())
    p = r[0]['spreadProfilePrices']
    pr = next((x for x in p if x['spreadProfile'] == 'prime'), p[0])
    return pd.Timestamp(r[0]['ts'], unit='ms', tz='UTC'), pr['bid'], pr['ask']


# ---------------------------------------------------------------- Mensajes
def ico(d):
    return {1: '🟢 ALCISTA', -1: '🔴 BAJISTA', 0: '⚪ NEUTRAL'}[int(d)]


def msg_plan(e, titulo='☀️ XAUUSD · PLAN DEL DÍA'):
    est = {1: 'HH + HL (alcista)', -1: 'LH + LL (bajista)', 0: 'mixta'}[e['estructura']]
    lado = 'VENTAS' if e['bias'] < 0 else 'COMPRAS'
    l = [f"<b>{titulo}</b>",
         f"🗓 {hora_es(e['t'], '%a %d/%m · %H:%M')} (España) · 💰 Precio {px(e['precio'])}",
         SEP,
         f"<b>1️⃣ DIARIO</b>  →  <b>{ico(e['bias'])}</b>",
         f"     Máx. ayer {px(e['pdh'])} · Mín. ayer {px(e['pdl'])}",
         f"<b>2️⃣ HORARIO</b>  →  <b>{ico(e['h1'])}</b>",
         f"     ADX {e['adx']:.1f} {'✅ con fuerza' if e['adx_ok'] else '⚠️ sin fuerza (necesita &gt; 20)'}",
         f"     Estructura {est}",
         f"     Resistencia {px(e['res'])} · Soporte {px(e['sop'])}"]
    if e['pauta']:
        l += [f"<b>3️⃣ PAUTA H1</b>",
              f"     Impulso {px(e['ori'])} ➜ {px(e['ext'])}",
              f"     🎯 Zona de {lado[:-1].lower()} (1/3 – 2/3): <b>{px(e['zona_a'])} – {px(e['zona_b'])}</b>",
              f"     Mitad (1/2): {px(e['fibo50'])} · Retroceso ahora: {e['retr_ahora']*100:.0f} %"]
    l.append(SEP)
    if e['bias'] == 0:
        plan = '⛔ Diario neutral: <b>hoy no se opera</b>.'
    elif e['h1'] != e['bias']:
        plan = '⏸ Diario y horario no coinciden: <b>esperar</b> a que se alineen.'
    elif not e['pauta']:
        plan = f'⏸ Solo {lado}, pero aún no hay impulso H1 válido: <b>esperar</b>.'
    elif e['retr'] > P['fmax']:
        plan = (f"⛔ El retroceso ya llega al {e['retr']*100:.0f} % (más de 2/3): <b>pauta invalidada</b>. "
                f"Esperar a que se forme un nuevo impulso {'bajista' if e['bias'] < 0 else 'alcista'} en H1.")
    elif e['preparado']:
        plan = f'🟠 <b>PREPARADO para {lado}</b>: el precio ya está en la zona. Vigilar el gatillo en 5 min.'
    elif not e['adx_ok']:
        plan = f'⏸ Solo {lado}. Falta fuerza (ADX ≤ 20): si llega a la zona, aún no vale.'
    else:
        plan = f'👀 Solo {lado}. <b>Esperar a que el precio llegue a la zona</b> y entonces al cruce EMA 9/21 + MACD en 5 min.'
    l += [f"📋 <b>PLAN:</b> {plan}",
          "⏰ Señales validadas 12:00 – 21:00 · Cierre obligatorio 22:00",
          "⚠️ Fuera de ese horario también te aviso, marcado como «no validado»."]
    return '\n'.join(l)


def msg_preparado(e, fuera=False):
    lado = 'VENTA' if e['bias'] < 0 else 'COMPRA'
    texto = '\n'.join([
        f"🟠 <b>PREPARADO · {lado} XAUUSD</b>",
        f"🕐 {hora_es(e['t'])} · 💰 Precio {px(e['precio'])}",
        SEP,
        f"El precio ha entrado en la zona óptima ({e['retr']*100:.0f} % de retroceso):",
        f"🎯 <b>{px(e['zona_a'])} – {px(e['zona_b'])}</b>",
        "",
        "👉 <b>Abre el gráfico de 5 minutos.</b> La entrada llega con el cruce de la EMA 9 "
        f"{'por debajo' if lado == 'VENTA' else 'por encima'} de la EMA 21 y el MACD a favor. "
        "Te avisaré con la señal y los niveles exactos."])
    return (AVISO_FUERA + texto) if fuera else texto


CERCA = 0.12   # aviso de nivel clave cuando el retroceso llega al 21 % (12 puntos antes del 1/3)


def pauta_formandose(win, h1, e):
    """Pauta H1 cuyo extremo aún no está confirmado (le faltan velas a la derecha): las mismas cuentas que
    M.estado, pero con el último máximo/mínimo provisional. La pauta oficial llega horas después,
    a menudo con el precio ya dentro de la zona. Se exige al menos 1 vela H1 cerrada sin extremo nuevo
    (si no, en plena caída avisaría cada hora)."""
    d = e['bias']; N = P['swing']
    if d == 0 or e['h1'] != d:
        return None
    tclose = e['t']; c = min(h1.index.searchsorted(tclose.floor('1h')), len(h1))
    h = h1.iloc[:c]; H, L = h.High.values, h.Low.values; X = L if d < 0 else H
    mejor = (lambda a, b: a < b) if d < 0 else (lambda a, b: a > b)
    k = next((i for i in range(c - 2, c - N - 1, -1) if i >= N and all(mejor(X[i], X[j]) or X[i] == X[j] for j in range(i - N, i))
              and all(mejor(X[i], X[j]) for j in range(i + 1, c))), None)
    if k is None or (e['pauta'] and e['ext_t'] == h.index[k]):
        return None
    sh, sl = M.swings(h, N); of = sh if d < 0 else sl
    iOri = next((i for i in range(k - 1, max(c - P['lookback'], 0), -1) if of[i]), None)
    if iOri is None:
        return None
    tr = pd.concat([h.High - h.Low, (h.High - h.Close.shift()).abs(), (h.Low - h.Close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    ext, ori = X[k], (H[iOri] if d < 0 else L[iOri]); rng = abs(ori - ext)
    seg = win.loc[h.index[k] + pd.Timedelta(hours=1):tclose - pd.Timedelta(minutes=1)]
    if rng < P['minimp'] * atr or (len(seg) and ((d < 0 and seg.Low.min() < ext) or (d > 0 and seg.High.max() > ext))):
        return None
    deep = (seg.High.max() if d < 0 else seg.Low.min()) if len(seg) else ext
    retr = abs(deep - ext) / rng
    if retr > P['fmax']:
        return None
    z1, z2 = ext - d * P['fmin'] * rng, ext - d * P['fmax'] * rng
    return dict(ext=ext, ori=ori, ext_t=h.index[k], retr=retr, retr_ahora=abs(e['precio'] - ext) / rng,
                zona_a=min(z1, z2), zona_b=max(z1, z2), oficial=h.index[k] + pd.Timedelta(hours=N + 1))


def _cuerpo_nivel(e, p, operadas):
    d = e['bias']; lado = 'VENTA' if d < 0 else 'COMPRA'
    inval = p['zona_b'] if d < 0 else p['zona_a']
    borde = p['zona_a'] if d < 0 else p['zona_b']
    dentro = p['zona_a'] <= e['precio'] <= p['zona_b']
    ok = lambda c: '✅' if c else '⬜'
    ini, fin = e['t'].normalize() + pd.Timedelta(hours=P['start']), e['t'].normalize() + pd.Timedelta(hours=P['end'])
    return [SEP,
            f"Impulso H1 {px(p['ori'])} ➜ {px(p['ext'])}",
            f"🎯 Zona de {lado.lower()}: <b>{px(p['zona_a'])} – {px(p['zona_b'])}</b>"
            + (' (el precio ya está dentro)' if dentro else f" (a {abs(e['precio'] - borde):.1f} $)"),
            f"⛔ Se anula si {'sube por encima' if d < 0 else 'baja por debajo'} de {px(inval)} (más de 2/3)",
            f"🛑 SL de B en el origen ±1 $ ≈ {px(p['ori'] - d * 1)} · los lotes exactos llegan con la SEÑAL",
            SEP, "<b>Qué tiene que cumplirse para entrar:</b>",
            f"✅ Diario {ico(d)}",
            f"✅ Horario {ico(e['h1'])} (EMA 20 {'&lt;' if d < 0 else '&gt;'} EMA 50)",
            f"{ok(e['adx_ok'])} ADX &gt; 20 en 1 hora (ahora {e['adx']:.1f})",
            f"{ok(dentro)} Precio dentro de la zona (retroceso ahora {p['retr_ahora']*100:.0f} %, hace falta 33–67 %)",
            f"{ok(en_horario(e['t']))} Hora de señales validadas ({hora_es(ini)}–{hora_es(fin)} España)",
            f"{ok(operadas == 0)} Sin operación v2.3 hoy (máximo 1 al día con el 3 %)",
            f"⬜ Gatillo en 5 min: EMA 9 cruza {'por debajo' if d < 0 else 'por encima'} de la EMA 21 con el "
            f"histograma MACD {'rojo' if d < 0 else 'verde'}, al cierre de la vela"]


def avisos_nivel(e, operadas=0, prov=None):
    """Avisos previos al PREPARADO, cada uno una vez por extremo H1 (claves = ext_t + 1/2/3 s, guardadas
    junto a las de los PREPARADO, que usan ext_t a secas):
    📐 pauta formándose (extremo sin confirmar) · 📐 pauta nueva · 📍 precio acercándose a la zona."""
    if e['bias'] == 0 or e['h1'] != e['bias']:
        return []
    lado = 'VENTA' if e['bias'] < 0 else 'COMPRA'
    cab = f"🕐 {hora_es(e['t'])} · 💰 Precio {px(e['precio'])}"
    out = []
    if prov:
        nuevo = 'mínimo' if e['bias'] < 0 else 'máximo'
        out.append((prov['ext_t'] + pd.Timedelta(seconds=3), '\n'.join([
            f"📐 <b>PAUTA FORMÁNDOSE · {lado} XAUUSD</b>", cab,
            f"<i>Provisional: el {nuevo} de las {hora_es(prov['ext_t'])} se confirma a las {hora_es(prov['oficial'])} "
            f"si no se hace un {nuevo} nuevo antes. Entonces llegará el PREPARADO.</i>",
            *_cuerpo_nivel(e, prov, operadas)])))
    if e['pauta'] and e['retr'] < P['fmin']:
        out.append((e['ext_t'] + pd.Timedelta(seconds=2),
                    '\n'.join([f"📐 <b>NUEVA PAUTA · {lado} XAUUSD</b>", cab, *_cuerpo_nivel(e, e, operadas)])))
        if e['retr_ahora'] >= P['fmin'] - CERCA:
            out.append((e['ext_t'] + pd.Timedelta(seconds=1),
                        '\n'.join([f"📍 <b>NIVEL CLAVE · {lado} XAUUSD</b> · el precio se acerca a la zona", cab,
                                   *_cuerpo_nivel(e, e, operadas)])))
    return out


CONTROLES = [(11, 0), (13, 0), (15, 35), (18, 0)]   # hora de España: apertura de la ventana, mediodía, tras abrir NY, tarde


def msg_control(e, prov=None, operadas=0):
    """Foto del oro a horas fijas, todos los días, para que Lucas valore por sí mismo si hay oportunidad."""
    d = e['bias']; p = e if e['pauta'] and e['retr'] <= P['fmax'] else prov
    a_favor = lambda x: '✅ a favor' if d and x * d > 0 else '❌ en contra' if d else '—'
    l = [f"📊 <b>PUNTO DE CONTROL · XAUUSD · {hora_es(e['t'])}</b>",
         f"💰 Precio {px(e['precio'])}",
         SEP,
         f"1️⃣ Diario {ico(d)} · 2️⃣ Horario {ico(e['h1'])} · ADX {e['adx']:.1f} {'✅' if e['adx_ok'] else '⚠️ (&lt; 20)'}",
         f"📏 Máx. ayer {px(e['pdh'])} · Mín. ayer {px(e['pdl'])}",
         f"📏 Resistencia H1 {px(e['res'])} · Soporte H1 {px(e['sop'])}",
         f"⏱ 5 min: EMA 9 {'&gt;' if e['ema_f'] > e['ema_s'] else '&lt;'} EMA 21 ({a_favor(e['ema_f'] - e['ema_s'])}) · "
         f"MACD {e['macd_hist']:+.2f} ({a_favor(e['macd_hist'])})"]
    if d == 0:
        l += [SEP, '⛔ Diario neutral: hoy la v2.3 no opera.']
    elif e['h1'] != d:
        l += [SEP, '⏸ Diario y horario no coinciden: la v2.3 espera a que se alineen.']
    elif p is None:
        if e['pauta']:
            l += [SEP, f"⛔ La pauta {px(e['ori'])} ➜ {px(e['ext'])} está anulada (retroceso {e['retr']*100:.0f} %). "
                       f"Esperar un nuevo {'mínimo' if d < 0 else 'máximo'} en 1 hora."]
        else:
            l += [SEP, '⏸ Aún no hay impulso válido en 1 hora: esperar.']
    else:
        estado = ('🟠 PREPARADO (el precio ya tocó la zona): vigilar el gatillo en 5 min' if e['preparado'] else
                  'provisional (extremo sin confirmar)' if p is prov else 'oficial')
        l += [f"📐 Pauta {estado}", *_cuerpo_nivel(e, p, operadas)]
    l += [SEP, "🔎 Divergencias (Trullás, 1, 5 y 15 min y 1 h): las avisa el vigía aparte.",
          "<i>Solo la 🚨 SEÑAL v2.3 va al 3 %; lo demás es para estar atento.</i>"]
    return '\n'.join(l)


def msg_senal(s, bal=CUENTA, fuera=False):
    d = s['dir']; lado = 'VENTA' if d < 0 else 'COMPRA'; col = '🔴' if d < 0 else '🟢'
    rA = usd(LOTE_A, abs(s['entry'] - s['slA'])); gA = usd(LOTE_A, abs(s['tpA'] - s['entry']))
    n = 2 if s['useB'] else 1
    l = [f"🚨 <b>SEÑAL · {lado} XAUUSD</b> {col}",
         f"🕐 {hora_es(s['t'] + pd.Timedelta(minutes=5))} · Entrada a mercado ≈ {px(s['entry'])}",
         SEP,
         f"📦 <b>ABRIR {n} POSICIÓN{'ES' if n > 1 else ''}</b>{' (mismo precio, distinto TP)' if n > 1 else ''}",
         "",
         f"🅰️ <b>Posición A · {LOTE_A:.2f} lotes</b> (TP cercano)",
         f"     ⛔ SL  {px(s['slA'])}   −{abs(s['entry']-s['slA']):.2f} $/oz → <b>−{rA:.0f} $</b>",
         f"     ✅ TP  {px(s['tpA'])}   +{abs(s['tpA']-s['entry']):.2f} $/oz → <b>+{gA:.0f} $</b>"]
    if s['useB']:
        rB = usd(LOTE_B, abs(s['entry'] - s['slB'])); gB = usd(LOTE_B, abs(s['tpB'] - s['entry']))
        l += ["",
              f"🅱️ <b>Posición B · {LOTE_B:.2f} lotes</b> (runner, stop lejano)",
              f"     ⛔ SL  {px(s['slB'])}   → <b>−{rB:.0f} $</b>",
              f"     ✅ TP  {px(s['tpB'])}   → <b>+{gB:.0f} $</b>  (extensión 1,618)"]
        riesgo = rA + rB
    else:
        l += ["", f"🅱️ Runner: <b>NO abrir</b> (su stop en {px(s['slB'])} supera el riesgo máximo del 4 %)."]
        riesgo = rA
    l += [SEP,
          f"⚖️ Riesgo total: <b>{riesgo:.0f} $</b> ({riesgo / bal * 100:.1f} % de la cuenta)",
          "🔁 Cuando A toque TP → " + (f"mueve el SL de B a la entrada {px(s['entry'])} (break-even)." if s['useB'] else 'operación terminada.'),
          "⏰ Cerrar todo a las 22:00 si sigue abierta.",
          "⚠️ Confirma los niveles en tu MT5 (el precio del broker puede variar unos céntimos)."]
    rf = FONDEO_EXTRA if fuera else FONDEO_RIESGO
    fA, fB = lotes_fondeo(s, rf)
    frA, fgA = usd(fA, abs(s['entry'] - s['slA'])), usd(fA, abs(s['tpA'] - s['entry']))
    l += [SEP,
          f"🏦 <b>CUENTA DE FONDEO · {FONDEO}</b> (riesgo {rf:.1%} = {FONDEO_CAP * rf:.0f} $"
          + (", operación extra no validada)" if fuera else ")"),
          f"🅰️ <b>{fA:.2f} lotes</b> · SL → <b>−{frA:.0f} $</b> · TP → <b>+{fgA:.0f} $</b>"]
    if fB:
        frB, fgB = usd(fB, abs(s['entry'] - s['slB'])), usd(fB, abs(s['tpB'] - s['entry']))
        l += [f"🅱️ <b>{fB:.2f} lotes</b> · SL → <b>−{frB:.0f} $</b> · TP → <b>+{fgB:.0f} $</b>",
              f"⚖️ Si salta todo: <b>−{frA + frB:.0f} $</b> ({(frA + frB) / FONDEO_CAP:.1%}) · "
              f"si A llega a TP y B a BE: <b>+{fgA:.0f} $</b> · si llegan los dos: <b>+{fgA + fgB:.0f} $</b>"]
    else:
        l += [f"⚖️ Si salta el SL: <b>−{frA:.0f} $</b> ({frA / FONDEO_CAP:.1%}) · si llega al TP: <b>+{fgA:.0f} $</b>"]
    l += [f"📏 Límites {FONDEO}: pérdida diaria máx. {FONDEO_CAP * 0.05:.0f} $ · total {FONDEO_CAP * 0.10:.0f} $. "
          + ("Tras 1 pérdida en el día, no operes más." if FONDEO_RIESGO >= 0.025 else "Tras 2 pérdidas en el día, no operes más.")]
    return (AVISO_FUERA if fuera else '') + '\n'.join(l)


# ---------------------------------------------------------------- Gestion al segundo
def gestionar(op, bid, ask, ahora):
    """Sigue la operacion avisada con el precio en vivo. Devuelve op o None al terminar."""
    if op is None:
        return None
    d = op['dir']; precio = bid if d > 0 else ask
    fA, fB = op.setdefault('lf', lotes_fondeo(op))
    f = lambda lotes, dist: f"  (fondeo: <b>{usd(lotes, dist):+.0f} $</b>)"
    def cierre(pos, nivel):
        op['res'] = op.get('res', 0) + usd(LOTE_A if pos == 'A' else LOTE_B, (nivel - op['entry']) * d)
    if ahora.hour >= P['close']:
        for pos in ('A', 'B'):
            if op[pos]:
                cierre(pos, precio); op[pos] = False
        tg(f"⏰ <b>CIERRE DIARIO · XAUUSD</b>\nSon las {hora_es(ahora)}: cierra a mercado lo que quede abierto ({px(precio)}).\n"
           f"Resultado estimado de la operación: <b>{op['res']:+.0f} $</b>")
        op['fin'] = ahora
        return None
    if op['A']:
        if (precio - op['slA']) * d <= 0:
            cierre('A', op['slA']); op['A'] = False
            tg(f"❌ <b>STOP A · XAUUSD</b>\nSe ha tocado el SL de A en {px(op['slA'])} → <b>{usd(LOTE_A, (op['slA']-op['entry'])*d):+.0f} $</b>" + f(fA, (op['slA']-op['entry'])*d)
               + ("\nEl runner B sigue abierto con su stop." if op['B'] else ''))
        elif (precio - op['tpA']) * d >= 0:
            cierre('A', op['tpA']); op['A'] = False
            if op['B']:
                op['slB'] = op['entry']
                tg(f"🎯 <b>TP A ALCANZADO · XAUUSD</b>\nA cerrada en {px(op['tpA'])} → <b>+{usd(LOTE_A, abs(op['tpA']-op['entry'])):.0f} $</b>"
                   + f(fA, abs(op['tpA']-op['entry'])) + "\n"
                   f"👉 <b>Mueve ahora el SL de B a {px(op['entry'])}</b> (break-even). Objetivo B: {px(op['tpB'])}")
            else:
                tg(f"🎯 <b>TP ALCANZADO · XAUUSD</b>\nPosición cerrada en {px(op['tpA'])} → <b>+{usd(LOTE_A, abs(op['tpA']-op['entry'])):.0f} $</b> ✅"
                   + f(fA, abs(op['tpA']-op['entry'])))
    if op['B']:
        if (precio - op['slB']) * d <= 0:
            cierre('B', op['slB']); op['B'] = False
            tg(f"ℹ️ <b>STOP B · XAUUSD</b>\nRunner cerrado en {px(op['slB'])} → <b>{usd(LOTE_B, (op['slB']-op['entry'])*d):+.0f} $</b>"
               + f(fB, (op['slB']-op['entry'])*d))
        elif (precio - op['tpB']) * d >= 0:
            cierre('B', op['tpB']); op['B'] = False
            tg(f"🏁 <b>OBJETIVO DEL RUNNER · XAUUSD</b>\nB cerrada en {px(op['tpB'])} → <b>+{usd(LOTE_B, abs(op['tpB']-op['entry'])):.0f} $</b> 🚀"
               + f(fB, abs(op['tpB']-op['entry'])))
    if not (op['A'] or op['B']):
        op['fin'] = ahora
        return None
    return op


# ---------------------------------------------------------------- Analisis y envios con grafico
def analizar(m1, ahora, PP=None):
    PP = PP or P
    win = m1[m1.index >= m1.index.max() - pd.Timedelta(days=200)]
    m5, h1, d1 = M.prep(win)
    m5 = m5[m5.index + pd.Timedelta(minutes=5) <= ahora.floor('5min')]
    t = M.run(win, m5, h1, d1, PP)
    e = M.estado(win, m5, h1, d1, PP)
    e['h1_velas'] = h1                                   # para pauta_formandose
    return win, m5, t, e


def en_horario(ts):
    return P['start'] <= ts.hour < P['end']


AVISO_FUERA = ('⚠️ <b>FUERA DE HORARIO · NO VALIDADO</b>\n'
               '<i>Se cumplen todas las reglas, pero fuera de 12:00–21:00. En el backtest estas entradas '
               'no fueron rentables (PF 1,01 frente a 1,55). Decides tú; si entras, mismo plan de gestión.</i>\n')


def enviar_plan(win, e, titulo='☀️ XAUUSD · PLAN DEL DÍA'):
    ruta = G.grafico_plan(win, e, os.path.join(IMG, 'plan.png'), P)
    tg(msg_plan(e, titulo), ruta, '📈 <b>Diario + horario</b>: EMAs, impulso, zona Fibonacci (turquesa) y máx./mín. de ayer (morado).')


def enviar_senal(win, e, s, fuera=False):
    ruta = G.grafico_senal(win, e, s, os.path.join(IMG, 'senal.png'), P)
    tg(msg_senal(s, fuera=fuera), ruta, '📈 <b>5 minutos</b>: entrada (azul), SL (rojo) y TP (verde).')


def enviar_cierre(win, e, ops, dia):
    tot = sum(o.get('res', 0) for o in ops)
    l = [f"🌙 <b>XAUUSD · CIERRE DEL DÍA</b>",
         f"🗓 {hora_es(dia, '%a %d/%m/%Y')} · Precio de cierre {px(e['precio'])}",
         SEP]
    if not ops:
        motivo = ('diario neutral' if e['bias'] == 0 else 'diario y horario no alineados' if e['h1'] != e['bias']
                  else 'el precio no llegó a la zona con fuerza' if not e['preparado'] else 'no hubo gatillo en 5 min')
        l.append(f"📭 Sin señales hoy ({motivo}). No operar también es operar bien.")
    for i, o in enumerate(ops, 1):
        lado = 'VENTA 🔴' if o['dir'] < 0 else 'COMPRA 🟢'
        l.append(f"{i}. {'⚠️ ' if o.get('fuera') else ''}{lado} a las {hora_es(o['t'] + pd.Timedelta(minutes=5))} en {px(o['entry'])} → <b>{o.get('res', 0):+.0f} $</b>"
                 + (' (fuera de horario)' if o.get('fuera') else ''))
    l += [SEP, f"💼 Resultado del día: <b>{tot:+.0f} $</b> ({tot / CUENTA * 100:+.1f} % de la cuenta)",
          f"🔭 Mañana: diario {ico(e['bias'])} · horario {ico(e['h1'])}"
          + (f" · zona {px(e['zona_a'])} – {px(e['zona_b'])}" if e['pauta'] else ''),
          "📝 Apunta la operación en tu diario de trading."]
    ruta = G.grafico_cierre(win, e, ops, dia, os.path.join(IMG, 'cierre.png'), P)
    tg('\n'.join(l), ruta, '📈 <b>Sesión del día</b> en 5 minutos con las entradas (▲/▼), SL y TP.')


# ---------------------------------------------------------------- Simulacion
def simular(fecha):
    global PREFIJO
    PREFIJO = f'SIMULACIÓN · día real {fecha} · cuenta {P["balance"]:.0f} $'
    m1 = pd.read_pickle(DATA).sort_index()
    dia = pd.Timestamp(fecha, tz='UTC')
    t_plan = pd.Timestamp(fecha + ' 08:00', tz=MADRID).tz_convert('UTC')
    win, _, _, e = analizar(m1[m1.index < t_plan], t_plan)
    enviar_plan(win, e)
    op = None; ops = []; prep = set()
    for T in pd.date_range(dia + pd.Timedelta(hours=P['start']), dia + pd.Timedelta(hours=P['end']), freq='5min'):
        win, m5, t, e = analizar(m1[m1.index < T], T + pd.Timedelta(seconds=10))
        ult = m5.index[-1]; s = t[t.t == ult] if len(t) else t
        if len(s):
            s = s.iloc[0].to_dict()
            enviar_senal(win, e, s)
            op = dict(t=s['t'], dir=s['dir'], entry=s['entry'], slA=s['slA'], tpA=s['tpA'], slB=s['slB'], tpB=s['tpB'],
                      A=True, B=bool(s['useB']), usoB=bool(s['useB']), res=0.0)
            ops.append(op)
            d = s['dir']
            for ts, row in m1.loc[T:dia + pd.Timedelta(hours=P['close'], minutes=1)].iterrows():
                malo, bueno = (row.Low, row.High) if d > 0 else (row.High, row.Low)
                if gestionar(op, malo, malo, ts) is None or gestionar(op, bueno, bueno, ts) is None:
                    break
            break
        if e['preparado'] and e['ext_t'] not in prep:
            prep.add(e['ext_t']); tg(msg_preparado(e))
    fin = dia + pd.Timedelta(hours=P['close'], minutes=5)
    win, _, _, e = analizar(m1[m1.index < fin], fin)
    enviar_cierre(win, e, ops, dia)


# ---------------------------------------------------------------- Bucle principal
_cafe = None


def mantener_despierto(ahora):
    global _cafe
    if not ES_MAC:
        return
    wd, h = ahora.weekday(), ahora.hour
    abierto = not (wd == 5 or (wd == 4 and h >= 21) or (wd == 6 and h < 22))
    if abierto and (_cafe is None or _cafe.poll() is not None):
        _cafe = subprocess.Popen(['caffeinate', '-ims', '-t', '1200'])


# ---------------------------------------------------------------- Estado entre ejecuciones (nube)
ESTADO = os.environ.get('XAU_ESTADO')          # si existe: se guarda/recupera el estado al parar y arrancar
DURACION = float(os.environ.get('XAU_DURACION_MIN', 0))   # minutos que corre antes de guardar y salir (0 = siempre)


def _ts(x):
    return pd.Timestamp(x) if x is not None else None


def guardar_estado(op, avisados_senal, avisados_prep, ops_dia, ultimo_plan, ultimo_cierre):
    if not ESTADO:
        return
    lista = lambda ops: [{k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in o.items()} for o in ops]
    est = dict(op=lista([op])[0] if op else None, avisados_senal=[str(x) for x in avisados_senal],
               avisados_prep=[str(x) for x in avisados_prep], ops_dia={str(k): lista(v) for k, v in ops_dia.items()},
               ultimo_plan=str(ultimo_plan) if ultimo_plan else None, ultimo_cierre=str(ultimo_cierre) if ultimo_cierre else None)
    os.makedirs(os.path.dirname(ESTADO) or '.', exist_ok=True)
    json.dump(est, open(ESTADO, 'w'), default=str)


def cargar_estado():
    if not ESTADO or not os.path.exists(ESTADO):
        return None
    est = json.load(open(ESTADO))
    conv = lambda o: {k: (_ts(v) if k in ('t', 'fin') else v) for k, v in o.items()}
    ops_dia = {pd.Timestamp(k).date(): [conv(o) for o in v] for k, v in est['ops_dia'].items()}
    op = conv(est['op']) if est['op'] else None
    if op:                                         # la operacion abierta es la misma que la del dia
        for lst in ops_dia.values():
            for i, o in enumerate(lst):
                if o.get('t') == op['t']:
                    lst[i] = op
    return dict(op=op, avisados_senal={_ts(x) for x in est['avisados_senal']},
                avisados_prep={_ts(x) for x in est['avisados_prep']}, ops_dia=ops_dia,
                ultimo_plan=pd.Timestamp(est['ultimo_plan']).date() if est['ultimo_plan'] else None,
                ultimo_cierre=pd.Timestamp(est['ultimo_cierre']).date() if est['ultimo_cierre'] else None)


def main():
    if '--setup' in sys.argv:
        return setup()
    if '--simular' in sys.argv:
        i = sys.argv.index('--simular')
        return simular(sys.argv[i + 1] if len(sys.argv) > i + 1 else '2026-07-13')
    m1 = completar_historico(pd.read_pickle(DATA)) if os.path.exists(DATA) else historico_inicial()
    ahora = pd.Timestamp.now(tz='UTC')
    win, m5, _, e = analizar(m1, ahora)
    if '--prueba' in sys.argv:
        return enviar_plan(win, e, '🧪 XAUUSD · PRUEBA DEL PLAN DEL DÍA')
    previo = cargar_estado()
    if previo is None:
        tg(f"🟢 <b>Monitor XAU en marcha</b> · precio en vivo cada segundo\n"
           f"Diario {ico(e['bias'])} · Horario {ico(e['h1'])} · Plan del día a las 08:00.")
    vela = None; ultima_eval = None; op = None
    avisados_senal, avisados_prep = set(), set()
    ops_dia = {}; ultimo_plan = None; ultimo_cierre = None
    if previo:
        op, avisados_senal, avisados_prep = previo['op'], previo['avisados_senal'], previo['avisados_prep']
        ops_dia, ultimo_plan, ultimo_cierre = previo['ops_dia'], previo['ultimo_plan'], previo['ultimo_cierre']
        log(f"Estado recuperado: operación abierta {'sí' if op else 'no'}, plan {ultimo_plan}, cierre {ultimo_cierre}")
    ultimo_guardado = time.time(); ultima_hora_completada = ahora.floor('1h')
    fin = time.time() + DURACION * 60 if DURACION else None
    while True:
        t0 = time.time()
        if fin and t0 > fin:
            m1 = m1.sort_index()
            m1[m1.index >= m1.index.max() - pd.Timedelta(days=220)].to_pickle(DATA)
            guardar_estado(op, avisados_senal, avisados_prep, ops_dia, ultimo_plan, ultimo_cierre)
            log('Fin del turno: estado y datos guardados.')
            return
        try:
            qt, bid, ask = cotizacion()
            ahora = pd.Timestamp.now(tz='UTC')
            mantener_despierto(ahora)
            fresca = (ahora - qt) < pd.Timedelta(minutes=3)
            if fresca:
                mnt = qt.floor('1min')
                if vela is not None and mnt > vela['t']:
                    m1.loc[vela['t'], ['Open', 'High', 'Low', 'Close', 'Vol']] = [vela['o'], vela['h'], vela['l'], vela['c'], vela['n']]
                if vela is None or mnt > vela['t']:
                    vela = dict(t=mnt, o=bid, h=bid, l=bid, c=bid, n=1)
                else:
                    vela.update(h=max(vela['h'], bid), l=min(vela['l'], bid), c=bid, n=vela['n'] + 1)
                op = gestionar(op, bid, ask, ahora)

            if ahora.floor('1h') > ultima_hora_completada and ahora.minute >= 15:
                ultima_hora_completada = ahora.floor('1h')
                m1 = completar_historico(m1.sort_index(), horas_atras=3)
            if time.time() - ultimo_guardado > 1800:
                m1 = m1.sort_index()
                m1[m1.index >= m1.index.max() - pd.Timedelta(days=220)].to_pickle(DATA)
                guardar_estado(op, avisados_senal, avisados_prep, ops_dia, ultimo_plan, ultimo_cierre)
                ultimo_guardado = time.time()

            laborable = ahora.weekday() < 5
            local = ahora.tz_convert(MADRID)
            if laborable and fresca and local.hour >= HORA_PLAN and local.hour < 12 and ultimo_plan != local.date():
                ultimo_plan = local.date()
                m1 = m1.sort_index(); win, _, _, e = analizar(m1, ahora); enviar_plan(win, e)
                if e['pauta']:
                    avisados_prep.add(e['ext_t'] + pd.Timedelta(seconds=2))   # el plan ya la cuenta
            if laborable and ahora.hour == P['close'] and ahora.minute >= 5 and ultimo_cierre != ahora.date():
                ultimo_cierre = ahora.date()
                m1 = m1.sort_index(); win, _, _, e = analizar(m1, ahora)
                enviar_cierre(win, e, ops_dia.get(ahora.date(), []), ahora.normalize())

            for hh, mm in CONTROLES:                       # puntos de control diarios (clave = hora + 4 s)
                tc = pd.Timestamp(local.date()).tz_localize(MADRID).replace(hour=hh, minute=mm).tz_convert('UTC') + pd.Timedelta(seconds=4)
                if laborable and fresca and tc <= ahora < tc + pd.Timedelta(minutes=30) and tc not in avisados_prep:
                    avisados_prep.add(tc)
                    try:
                        m1 = m1.sort_index(); win, _, _, e = analizar(m1, ahora)
                        tg(msg_control(e, pauta_formandose(win, e['h1_velas'], e), len(ops_dia.get(ahora.date(), []))))
                    except Exception:
                        log('Error en punto de control: ' + traceback.format_exc().splitlines()[-1])
            barra = ahora.floor('5min') - pd.Timedelta(minutes=5)
            if fresca and barra != ultima_eval and (ahora - ahora.floor('5min')) >= pd.Timedelta(seconds=3):
                ultima_eval = barra
                m1 = m1.sort_index()
                win, m5, t, e = analizar(m1, ahora)
                ult = m5.index[-1]
                s = t[t.t == ult] if len(t) else t
                if len(s) and ult not in avisados_senal:
                    s = s.iloc[0].to_dict(); avisados_senal.add(ult)
                    if op is None:
                        enviar_senal(win, e, s)
                        op = dict(t=s['t'], dir=s['dir'], entry=s['entry'], slA=s['slA'], tpA=s['tpA'], slB=s['slB'],
                                  tpB=s['tpB'], A=True, B=bool(s['useB']), usoB=bool(s['useB']), res=0.0)
                        ops_dia.setdefault(ahora.date(), []).append(op)
                    else:
                        tg('ℹ️ Nueva señal XAUUSD ignorada: ya hay una operación abierta.')
                elif op is None and e['preparado'] and e['ext_t'] not in avisados_prep:
                    avisados_prep.add(e['ext_t']); tg(msg_preparado(e))
                elif op is None and not en_horario(ahora - pd.Timedelta(seconds=10)):
                    # Mismas reglas sin restriccion horaria: aviso marcado como no validado
                    win2, m5b, tl, el = analizar(m1, ahora, P_LIBRE)
                    sl_ = tl[tl.t == ult] if len(tl) else tl
                    hoy = ops_dia.get(ahora.date(), [])
                    if len(sl_) and ult not in avisados_senal and len(hoy) < P['maxday']:
                        s = sl_.iloc[0].to_dict(); avisados_senal.add(ult)
                        enviar_senal(win2, el, s, fuera=True)
                        op = dict(t=s['t'], dir=s['dir'], entry=s['entry'], slA=s['slA'], tpA=s['tpA'], slB=s['slB'],
                                  tpB=s['tpB'], A=True, B=bool(s['useB']), usoB=bool(s['useB']), res=0.0, fuera=True)
                        ops_dia.setdefault(ahora.date(), []).append(op)
                    elif el['preparado'] and el['ext_t'] not in avisados_prep:
                        avisados_prep.add(el['ext_t']); tg(msg_preparado(el, fuera=True))
                if op is None and laborable and local.hour >= HORA_PLAN and ahora.hour < P['end']:
                    try:                                   # avisos previos: un fallo aquí no debe frenar el monitor
                        for clave, texto in avisos_nivel(e, len(ops_dia.get(ahora.date(), [])), pauta_formandose(win, e['h1_velas'], e)):
                            if clave not in avisados_prep:
                                avisados_prep.add(clave); tg(texto)
                    except Exception:
                        log('Error en avisos de nivel: ' + traceback.format_exc().splitlines()[-1])
                log(f"M5 {hora_es(ult + pd.Timedelta(minutes=5))} precio {e['precio']:.2f} | diario {e['bias']} h1 {e['h1']} "
                    f"adx {e['adx']:.1f} | pauta {e['pauta']} zona {e['zona']} preparado {e['preparado']}")
        except KeyboardInterrupt:
            raise
        except Exception:
            log('Error: ' + traceback.format_exc().splitlines()[-1])
        time.sleep(max(0.2, 1.0 - (time.time() - t0)))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log('Monitor detenido.')
