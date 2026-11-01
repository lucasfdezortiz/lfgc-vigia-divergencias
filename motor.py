# Motor de senales XAU Fractal DayTrader (mismo codigo que el backtest).
import pandas as pd, numpy as np, sys, os
def load(path=None):
    if path and os.path.exists(path): m1=pd.read_pickle(path)
    else:
        fs=sorted(f for f in os.listdir('duk') if os.path.getsize('duk/'+f)>1000)
        m1=pd.concat([pd.read_pickle('duk/'+f) for f in fs]); m1=m1[m1.Vol>0]
    m1=m1.rename(columns={'Volume':'Vol'})
    return m1[['Open','High','Low','Close','Vol']]
agg=dict(Open='first',High='max',Low='min',Close='last',Vol='sum')
def ema(s,n): return s.ewm(span=n,adjust=False).mean()

def macd(c,f=12,s=26,g=9):
    line=ema(c,f)-ema(c,s); sig=ema(line,g); return line,sig,line-sig
def rsi(c,n=14):
    d=c.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+up/dn)
def adx(df,n=14):
    up=df.High.diff(); dn=-df.Low.diff()
    pdm=np.where((up>dn)&(up>0),up,0.0); ndm=np.where((dn>up)&(dn>0),dn,0.0)
    tr=pd.concat([df.High-df.Low,(df.High-df.Close.shift()).abs(),(df.Low-df.Close.shift()).abs()],axis=1).max(axis=1)
    a=tr.ewm(alpha=1/n,adjust=False).mean()
    pdi=100*pd.Series(pdm,df.index).ewm(alpha=1/n,adjust=False).mean()/a
    ndi=100*pd.Series(ndm,df.index).ewm(alpha=1/n,adjust=False).mean()/a
    dx=100*(pdi-ndi).abs()/(pdi+ndi); return dx.ewm(alpha=1/n,adjust=False).mean()

def prep(m1):
    m5=m1.resample('5min').agg(agg).dropna()
    m5=m5[m5.Vol>0] if (m5.Vol>0).mean()>0.5 else m5
    h1=m1.resample('1h').agg(agg).dropna()
    d1=m1.resample('D',offset='-2h').agg(agg).dropna()   # dia de broker (22:00 UTC)
    d1=d1[(m1.resample('D',offset='-2h').Close.count()>300).reindex(d1.index).fillna(False)]
    return m5,h1,d1

def run(m1,m5,h1,d1,P):
    N=P['swing']; spread=P['cost']
    # D1 sesgo del dia anterior
    d1=d1.copy(); d1['f']=ema(d1.Close,P['d1f']); d1['s']=ema(d1.Close,P['d1s'])
    d1['bias']=np.where((d1.Close>d1.s)&(d1.f>d1.s),1,np.where((d1.Close<d1.s)&(d1.f<d1.s),-1,0))
    d1['bias']=d1.bias.shift(1); d1['adx']=adx(d1).shift(1)
    if P.get('nobias'): d1['bias']=np.nan
    h=h1.copy(); h['f']=ema(h.Close,P['h1f']); h['s']=ema(h.Close,P['h1s'])
    tr=pd.concat([h.High-h.Low,(h.High-h.Close.shift()).abs(),(h.Low-h.Close.shift()).abs()],axis=1).max(axis=1)
    h['atr']=tr.rolling(14).mean(); _,_,h['mh']=macd(h.Close); h['adx']=adx(h)
    H=h.High.values; L=h.Low.values; n=len(h)
    sh=np.zeros(n,bool); sl=np.zeros(n,bool)
    for i in range(N,n-N):
        w=H[i-N:i+N+1]; sh[i]=H[i]==w.max() and (H[i+1:i+N+1]<H[i]).all() and (H[i-N:i]<=H[i]).all()
        w=L[i-N:i+N+1]; sl[i]=L[i]==w.min() and (L[i+1:i+N+1]>L[i]).all() and (L[i-N:i]>=L[i]).all()
    hidx=h.index
    m=(m1.resample(P['tf']).agg(agg).dropna() if P.get('tf') else m5).copy(); m['f']=ema(m.Close,P['m5f']); m['s']=ema(m.Close,P['m5s'])
    tfmin=int(pd.Timedelta(P.get('tf','5min')).total_seconds()//60)
    ml,ms,mh=macd(m.Close); m['mh']=mh; m['rsi']=rsi(m.Close)
    if P.get('trigger')=='macd':
        up=(ml>ms)&(ml.shift()<=ms.shift()); dn=(ml<ms)&(ml.shift()>=ms.shift())
    elif P.get('trigger')=='macd0':
        up=(mh>0)&(mh.shift()<=0)&(ml<0); dn=(mh<0)&(mh.shift()>=0)&(ml>0)
    else:
        up=(m.f>m.s)&(m.f.shift()<=m.s.shift()); dn=(m.f<m.s)&(m.f.shift()>=m.s.shift())
    ML=ml.values; MH=m.mh.values; RS=m.rsi.values; O5=m.Open.values; C5=m.Close.values; H5=m.High.values; L5=m.Low.values
    dkey=(m.index+pd.Timedelta(hours=2)).normalize()-pd.Timedelta(hours=2)
    m1H=m1.High.values; m1L=m1.Low.values; m1C=m1.Close.values; m1t=m1.index
    trades=[]; busy_until=None; per_day={}; last_ext=None; bal=P['balance']
    sig=np.where(up,1,np.where(dn,-1,0))
    for k in np.nonzero(sig)[0]:
        t=m.index[k]; tclose=t+pd.Timedelta(minutes=tfmin)
        if busy_until is not None and tclose<busy_until: continue
        if not (P['start']<=tclose.hour<P['end']): continue
        dk=dkey[k]
        if per_day.get(dk,0)>=P['maxday']: continue
        dir_=sig[k]
        b=d1.bias.get(dk,np.nan)
        if not P.get('nobias'):
            if pd.isna(b) or b!=dir_: continue
        c=hidx.searchsorted(tclose.floor('1h'))           # indice de la vela H1 en formacion
        if c<60: continue
        if P.get('adx_d1') and not (d1.adx.get(dk,0)>P['adx_d1']): continue
        if P.get('adx_h1') and not (h.adx.iloc[c-1]>P['adx_h1']): continue
        if P.get('macd_m5') and not (MH[k]*dir_>0): continue
        if P.get('macd_h1') and not ((h.mh.iloc[c-1]-h.mh.iloc[c-2])*dir_>0): continue
        if P.get('rsi_m5') and not ((RS[k]-50)*dir_>0): continue
        if P.get('candle') and not ((C5[k]-O5[k])*dir_>0 and ((C5[k]>H5[k-1]) if dir_>0 else (C5[k]<L5[k-1]))): continue
        if not P.get('noh1trend'):
            hf,hs=h.f.iloc[c-1],h.s.iloc[c-1]
            if (dir_>0 and hf<=hs) or (dir_<0 and hf>=hs): continue
        ext_flag=sh if dir_>0 else sl; ori_flag=sl if dir_>0 else sh
        iExt=None
        for i in range(c-1-N,max(c-P['lookback'],0),-1):
            if ext_flag[i]: iExt=i; break
        if iExt is None: continue
        iOri=None
        for i in range(iExt-1,max(c-P['lookback'],0),-1):
            if ori_flag[i]: iOri=i; break
        if iOri is None: continue
        ext=H[iExt] if dir_>0 else L[iExt]; ori=L[iOri] if dir_>0 else H[iOri]
        # precio desde el extremo hasta el cierre de la vela M5 de senal
        seg=m1.loc[hidx[iExt]+pd.Timedelta(hours=1):tclose-pd.Timedelta(minutes=1)]
        if seg.empty: continue
        if dir_>0:
            if seg.High.max()>ext: continue
            deep=min(seg.Low.min(),np.inf)
        else:
            if seg.Low.min()<ext: continue
            deep=seg.High.max()
        rng=abs(ext-ori)
        if rng<P['minimp']*h.atr.iloc[c-1]: continue
        if hidx[iExt]==last_ext: continue
        retr=(ext-deep)/rng if dir_>0 else (deep-ext)/rng
        if not P.get('nofib') and not (P['fmin']<=retr<=P['fmax']): continue
        # --- Consejos Cava/Trullas: agotamiento del retroceso ---
        conf=0
        pb=m.loc[hidx[iExt]+pd.Timedelta(hours=1):t]
        imp=m.loc[hidx[iOri]:hidx[iExt]+pd.Timedelta(hours=1)]
        volpb = len(pb)>3 and len(imp)>3 and imp.Vol.mean()>0 and pb.Vol.mean() < P.get('volratio',1.0)*imp.Vol.mean()
        div=False; rdiv=False
        if len(pb)>6:
            ml_pb=ML[m.index.get_indexer(pb.index)]; rs_pb=RS[m.index.get_indexer(pb.index)]
            lows=(pb.Low.values if dir_>0 else -pb.High.values)
            p=int(np.argmin(lows))
            if p>=3:
                if dir_>0: div = ml_pb[p] > ml_pb[:p].min()+1e-9; rdiv = rs_pb[p] > rs_pb[:p].min()+1e-9
                else: div = ml_pb[p] < ml_pb[:p].max()-1e-9; rdiv = rs_pb[p] < rs_pb[:p].max()-1e-9
        conf = int(volpb)+int(div)+int(rdiv)
        if P.get('need_vol') and not volpb: continue
        if P.get('need_div') and not (div or rdiv): continue
        if P.get('need_conf',0) and conf<P['need_conf']: continue
        entry=m.Close.iloc[k]
        if (dir_>0 and entry>=ext) or (dir_<0 and entry<=ext): continue
        slA=deep-dir_*P['buf']; riskA=abs(entry-slA)
        if riskA<=0: continue
        tpA=entry+dir_*P['rrA']*riskA if P['tpA']=='rr' else ext
        if P['tpA']!='rr' and abs(tpA-entry)<0.3*riskA: continue
        slB=ori-dir_*P['buf']; tpB=ori+dir_*P['ext']*rng
        maxr=bal*P['maxrisk']/100
        rA=P['lotA']*100*riskA; rB=P['lotB']*100*abs(entry-slB)
        last_ext=hidx[iExt]
        if rA>maxr: continue
        useB=P['lotB']>0 and rA+rB<=maxr and (not P.get('b_if_conf') or conf>=P['b_if_conf'])
        # simulacion con M1
        j0=m1t.searchsorted(tclose)
        endt=tclose.normalize()+pd.Timedelta(hours=P['close'])
        if endt<=tclose: continue
        openA=True; openB=useB; pA=pB=0.0; slBcur=slB; tend=None
        j=j0
        while j<len(m1t) and (openA or openB):
            tt=m1t[j]
            if tt>=endt:
                px=m1.Open.iloc[j]
                if openA: pA=(px-entry)*dir_; openA=False
                if openB: pB=(px-entry)*dir_; openB=False
                tend=tt; break
            hi,lo=m1H[j],m1L[j]
            adv = lo if dir_>0 else hi; fav = hi if dir_>0 else lo
            if openA:
                if (adv-slA)*dir_<=0: pA=(slA-entry)*dir_; openA=False
                elif (fav-tpA)*dir_>=0:
                    pA=(tpA-entry)*dir_; openA=False
                    if P['be'] and openB: slBcur=entry
            if openB and P.get('trail') and not openA and pA>0:
                w=m1L[max(j-P['trail'],j0):j+1] if dir_>0 else m1H[max(j-P['trail'],j0):j+1]
                cand=(w.min()-P['buf']) if dir_>0 else (w.max()+P['buf'])
                slBcur=max(slBcur,cand) if dir_>0 else min(slBcur,cand)
            if openB:
                if (adv-slBcur)*dir_<=0: pB=(slBcur-entry)*dir_; openB=False
                elif (fav-tpB)*dir_>=0: pB=(tpB-entry)*dir_; openB=False
                elif P['be'] and (fav-tpA)*dir_>=0: slBcur=entry
            tend=tt; j+=1
        usd=P['lotA']*100*(pA-spread)+(P['lotB']*100*(pB-spread) if useB else 0)
        bal+=usd
        per_day[dk]=per_day.get(dk,0)+1; busy_until=tend
        trades.append(dict(entry=entry,slA=slA,tpA=tpA,slB=slB,tpB=tpB,ext=ext,ori=ori,rA=rA,rB=rB,t=t,dir=dir_,conf=conf,volpb=volpb,div=div,rdiv=rdiv,retr=retr,riskA=riskA,riskUSD=rA+(rB if useB else 0),useB=useB,
                           pA=pA-spread,pB=(pB-spread) if useB else np.nan,usd=usd,bal=bal))
    return pd.DataFrame(trades)

BASE=dict(swing=3,lookback=120,d1f=20,d1s=50,h1f=20,h1s=50,m5f=9,m5s=21,minimp=1.5,fmin=0.333,fmax=0.667,
          buf=1.0,rrA=1.0,tpA='rr',ext=1.618,be=True,lotA=0.02,lotB=0.01,maxrisk=4.0,maxday=2,
          start=7,end=19,close=20,cost=0.40,balance=820.0)

def summary(name,t,bal0=820.0):
    if t.empty: return print(f"{name:38s} sin operaciones")
    eq=bal0+t.usd.cumsum(); dd=((eq-eq.cummax())/eq.cummax()).min()
    w=t.usd>0; pf=t.usd[w].sum()/-t.usd[~w].sum() if (~w).any() else np.inf
    yrs=" ".join(f"{y}:{g.usd.sum():+.0f}" for y,g in t.groupby(t.t.dt.year))
    print(f"{name:38s} n={len(t):4d} acierto={w.mean():.0%} PF={pf:.2f} total=${t.usd.sum():+7.0f} "
          f"medio=${t.usd.mean():+.2f} maxDD={dd:.0%} B usado={t.useB.mean():.0%} | {yrs}")


# Configuracion validada v2.3 (backtest ene-2023 -> oct-2026)
V23=dict(BASE); V23.update(macd_m5=True, adx_h1=20, start=10, end=19, close=20)

def swings(h,N):
    H=h.High.values; L=h.Low.values; n=len(h)
    sh=np.zeros(n,bool); sl=np.zeros(n,bool)
    for i in range(N,n-N):
        w=H[i-N:i+N+1]; sh[i]=H[i]==w.max() and (H[i+1:i+N+1]<H[i]).all() and (H[i-N:i]<=H[i]).all()
        w=L[i-N:i+N+1]; sl[i]=L[i]==w.min() and (L[i+1:i+N+1]>L[i]).all() and (L[i-N:i]>=L[i]).all()
    return sh,sl

def estado(m1,m5,h1,d1,P=V23):
    """Foto del mercado al cierre de la ultima vela M5: Diario -> Horario -> 5 minutos."""
    N=P['swing']; t=m5.index[-1]; tclose=t+pd.Timedelta(minutes=5)
    d=d1.copy(); d['f']=ema(d.Close,P['d1f']); d['s']=ema(d.Close,P['d1s'])
    d['bias']=np.where((d.Close>d.s)&(d.f>d.s),1,np.where((d.Close<d.s)&(d.f<d.s),-1,0)); d['bias']=d.bias.shift(1)
    dk=(tclose-pd.Timedelta(minutes=5)+pd.Timedelta(hours=2)).normalize()-pd.Timedelta(hours=2)
    bias=d.bias.get(dk,np.nan)
    if pd.isna(bias): bias=int(d.bias.iloc[-1]) if not pd.isna(d.bias.iloc[-1]) else 0
    prev=d[d.index<dk].iloc[-1] if (d.index<dk).any() else d.iloc[-1]
    h=h1.copy(); h['f']=ema(h.Close,P['h1f']); h['s']=ema(h.Close,P['h1s'])
    tr=pd.concat([h.High-h.Low,(h.High-h.Close.shift()).abs(),(h.Low-h.Close.shift()).abs()],axis=1).max(axis=1)
    h['atr']=tr.rolling(14).mean(); h['adx']=adx(h)
    hidx=h.index; c=hidx.searchsorted(tclose.floor('1h')); c=min(c,len(h))
    sh,sl=swings(h,N)
    hf,hs=h.f.iloc[c-1],h.s.iloc[c-1]; h1t=1 if hf>hs else -1 if hf<hs else 0
    adxv=h.adx.iloc[c-1]; atr=h.atr.iloc[c-1]
    idx_h=[i for i in range(c-1-N,0,-1) if sh[i]][:2]; idx_l=[i for i in range(c-1-N,0,-1) if sl[i]][:2]
    est=1 if len(idx_h)==2 and len(idx_l)==2 and h.High.iloc[idx_h[0]]>h.High.iloc[idx_h[1]] and h.Low.iloc[idx_l[0]]>h.Low.iloc[idx_l[1]] else \
        -1 if len(idx_h)==2 and len(idx_l)==2 and h.High.iloc[idx_h[0]]<h.High.iloc[idx_h[1]] and h.Low.iloc[idx_l[0]]<h.Low.iloc[idx_l[1]] else 0
    out=dict(t=tclose,precio=m5.Close.iloc[-1],bias=int(bias),h1=h1t,adx=adxv,adx_ok=adxv>P['adx_h1'],estructura=est,
             pdh=prev.High,pdl=prev.Low,res=h.High.iloc[idx_h[0]] if idx_h else np.nan,sop=h.Low.iloc[idx_l[0]] if idx_l else np.nan,
             pauta=False,zona=False,preparado=False)
    m=m5.copy(); _,_,mh=macd(m.Close); out['macd_hist']=mh.iloc[-1]; out['rsi']=rsi(m.Close).iloc[-1]
    out['ema_f']=ema(m.Close,P['m5f']).iloc[-1]; out['ema_s']=ema(m.Close,P['m5s']).iloc[-1]
    dir_=out['bias']
    if dir_==0 or h1t!=dir_: return out
    ef=sh if dir_>0 else sl; of=sl if dir_>0 else sh
    iExt=next((i for i in range(c-1-N,max(c-P['lookback'],0),-1) if ef[i]),None)
    if iExt is None: return out
    iOri=next((i for i in range(iExt-1,max(c-P['lookback'],0),-1) if of[i]),None)
    if iOri is None: return out
    H=h.High.values; L=h.Low.values
    ext=H[iExt] if dir_>0 else L[iExt]; ori=L[iOri] if dir_>0 else H[iOri]; rng=abs(ext-ori)
    seg=m1.loc[hidx[iExt]+pd.Timedelta(hours=1):tclose-pd.Timedelta(minutes=1)]
    if seg.empty or rng<P['minimp']*atr: return out
    if (dir_>0 and seg.High.max()>ext) or (dir_<0 and seg.Low.min()<ext): return out
    deep=seg.Low.min() if dir_>0 else seg.High.max()
    retr=(ext-deep)/rng if dir_>0 else (deep-ext)/rng
    ahora=(ext-out['precio'])/rng if dir_>0 else (out['precio']-ext)/rng
    z1=ext-dir_*P['fmin']*rng; z2=ext-dir_*P['fmax']*rng
    out.update(pauta=True,ext=ext,ori=ori,ext_t=hidx[iExt],retr=retr,retr_ahora=ahora,zona_a=min(z1,z2),zona_b=max(z1,z2),
               fibo50=ext-dir_*0.5*rng,zona=P['fmin']<=retr<=P['fmax'])
    out['preparado']=out['zona'] and out['adx_ok'] and P['start']<=tclose.hour<P['end']
    return out
