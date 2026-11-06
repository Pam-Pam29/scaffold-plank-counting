"""Line-based variant, like the reference app: user drags a vertical line over the stack; count along that line only."""
import cv2, numpy as np
from baseline import ROIS, count_layers
from dp_counter import segment
from scipy.signal import savgol_filter
TRUTH={'img1':57,'img2':23,'img3':72,'img4':15,'img5':60}
def line_profile(gray, roi, frac=0.12):
    x0,x1,y0,y1=roi; cx=(x0+x1)//2; hw=max(2,int((x1-x0)*frac/2))
    c=cv2.GaussianBlur(gray[y0:y1,cx-hw:cx+hw].astype(np.float32),(0,0),1.0)
    p=np.abs(cv2.Sobel(c,cv2.CV_32F,0,1,ksize=3)).mean(axis=1)
    return savgol_filter(p,5,2)
for name,rois in ROIS.items():
    g=cv2.cvtColor(cv2.imread(f"images/{name}.png"),cv2.COLOR_BGR2GRAY); A=[];B=[]
    for r in rois:
        p=line_profile(g,r); npk,_,per,_=count_layers(p); per=per or 20
        A.append(int(npk)); B.append(len(segment(p,pmin=max(8,int(0.62*per)),pmax=int(1.7*per)))+1)
    print(name,"truth",TRUTH[name],"| line+peaks",A,sum(A),"| line+DP",B,sum(B))
