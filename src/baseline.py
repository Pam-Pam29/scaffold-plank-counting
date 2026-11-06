"""Baseline plank-layer counter: vertical edge profile + periodicity, per stack ROI (no training)."""
import cv2, numpy as np, json, sys, time
from scipy.signal import find_peaks, savgol_filter

# ROIs per stack: (x0, x1, y0, y1) on the END FACES. Semi-manual; same idea as a user tapping a stack in the app.
ROIS = {
 "img1": [(0,85,230,447),(90,185,225,447),(188,285,215,447),(290,400,205,447)],
 "img2": [(5,120,235,447),(125,250,250,447),(240,350,285,447)],
 "img3": [(60,135,60,365),(140,218,60,365),(222,295,60,365),(300,378,60,365)],
 "img4": [(0,200,535,700),(205,480,510,700),(485,700,540,700)],
 "img5": [(10,150,35,735),(160,310,20,765),(325,480,10,780)],
}

def profile(gray, roi):
    x0,x1,y0,y1 = roi
    c = gray[y0:y1, x0:x1].astype(np.float32)
    c = cv2.GaussianBlur(c,(0,0),1.0)
    gy = np.abs(cv2.Sobel(c, cv2.CV_32F, 0, 1, ksize=3))
    p = gy.mean(axis=1)                       # row-wise horizontal-edge strength
    p = savgol_filter(p, 5, 2)
    return p

def count_layers(p):
    h = len(p)
    pn = (p - p.mean())
    ac = np.correlate(pn, pn, 'full')[h-1:]
    ac /= ac[0]
    # dominant period = first strong autocorr peak beyond lag 4
    pk,_ = find_peaks(ac[4:h//2], height=0.1)
    # pick the strongest autocorr peak (not the first): avoids locking onto half-pitch edges (top face + front edge of each plank)
    period = int(pk[np.argmax(ac[4:h//2][pk])]+4) if len(pk) else None
    dist = max(3, int(0.6*period)) if period else 6
    peaks,_ = find_peaks(p, distance=dist, prominence=0.25*np.std(p))
    n_peak = len(peaks)+1                      # boundaries + 1
    n_period = h/period if period else None
    return n_peak, n_period, period, peaks

if __name__ == "__main__":
    out = {}
    for name, rois in ROIS.items():
        img = cv2.imread(f"images/{name}.png"); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        vis = img.copy(); res=[]
        t=time.perf_counter()
        for i,r in enumerate(rois):
            p = profile(gray, r); npk, npd, per, peaks = count_layers(p)
            res.append(dict(stack=i+1, peak_count=int(npk), period_count=None if npd is None else round(float(npd),1), period_px=None if per is None else int(per)))
            x0,x1,y0,y1 = r
            cv2.rectangle(vis,(x0,y0),(x1,y1),(0,200,255),1)
            for y in peaks: cv2.line(vis,(x0,y0+int(y)),(x1,y0+int(y)),(0,0,255),1)
            cv2.putText(vis,str(npk),(x0+4,y0+16),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        ms=(time.perf_counter()-t)*1000
        cv2.imwrite(f"out_baseline_{name}.png", vis)
        out[name]=dict(stacks=res, ms=round(ms,1)); print(name, res, f"{ms:.1f} ms")
    json.dump(out, open("baseline_results.json","w"), indent=1)
