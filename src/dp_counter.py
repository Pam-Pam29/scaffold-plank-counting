"""Perspective-aware layer counter: dynamic programming over boundary positions.
Boundaries must fall on strong horizontal edges; plank height may shrink/grow smoothly (perspective)."""
import cv2, numpy as np, json, time
from baseline import ROIS, profile, count_layers

def segment(p, pmin=8, pmax=90, lam=6.0, tau=0.35, max_ratio=1.35, pstart=8):
    p = (p - np.median(p)) / (np.std(p) + 1e-6)          # normalised edge strength
    n = len(p); NEG = -1e18
    gain = p - tau                                         # boundary must beat threshold tau to be worth adding
    best = np.full((n, pmax+1), NEG); prev = {}
    for y in range(n):                                     # first boundary: free pitch from top (virtual boundary at 0)
        d = y
        if max(pmin,pstart) <= d <= pmax: best[y, d] = gain[y] - 0.0
    for y in range(n):
        for d in range(pmin, pmax+1):
            s = best[y, d]
            if s <= NEG/2: continue
            for d2 in range(max(pmin,int(d/max_ratio)), min(pmax,int(d*max_ratio))+1):
                y2 = y + d2
                if y2 >= n: break
                c = s + gain[y2] - lam*(np.log(d2/d))**2
                if c > best[y2, d2]: best[y2, d2] = c; prev[(y2,d2)] = (y,d)
    # end: choose best state; allow tail shorter than a pitch
    ys, ds = np.unravel_index(np.argmax(best), best.shape)
    path=[]; state=(ys,ds)
    while state in prev: path.append(state[0]); state=prev[state]
    path.append(state[0]); path=sorted(path)
    return path

def count(p, **kw):
    b = segment(p, **kw)
    return len(b)+1, b      # boundaries + 1 (last layer has no lower boundary in crop)

CONFIGS = {  # the three DP settings reported in the write-up
    "B1": lambda per: dict(),                                                       # loose: free plank height
    "B2": lambda per: dict(pmin=max(8,int(0.62*per)), pmax=int(1.7*per)),           # plank height tied to dominant period
    "B3": lambda per: dict(pmin=max(8,int(0.45*per)), pmax=int(1.7*per), pstart=int(0.8*per), lam=12.0),  # gradual change
}

if __name__=="__main__":
    import sys
    cfg = sys.argv[1] if len(sys.argv)>1 else "B2"
    res={}
    for name,rois in ROIS.items():
        img=cv2.imread(f"images/{name}.png"); g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); vis=img.copy(); r_=[]
        t=time.perf_counter()
        for r in rois:
            x0,x1,y0,y1=r; p=profile(g,r); per=count_layers(p)[2] or 20
            n,b=count(p,**CONFIGS[cfg](per)); r_.append(n)
            cv2.rectangle(vis,(x0,y0),(x1,y1),(0,200,255),1)
            for y in b: cv2.line(vis,(x0,y0+y),(x1,y0+y),(0,0,255),1)
            cv2.putText(vis,str(n),(x0+4,y0+16),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        ms=(time.perf_counter()-t)*1000; cv2.imwrite(f"out_dp_{cfg}_{name}.png",vis); res[name]=dict(counts=r_,total=sum(r_),ms=round(ms))
        print(name,r_,sum(r_),f"{ms:.0f} ms")
    json.dump(res,open(f"dp_results_{cfg}.json","w"),indent=1)
