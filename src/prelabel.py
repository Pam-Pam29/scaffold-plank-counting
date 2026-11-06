"""Pre-label plank end faces as YOLO boxes from the classical pipeline (method B2). Boxes are a starting point; correct them by hand."""
import cv2, os, shutil
from baseline import ROIS, profile, count_layers
from dp_counter import segment
OUT="scaffold_data"; shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(f"{OUT}/images"); os.makedirs(f"{OUT}/labels"); os.makedirs(f"{OUT}/preview")
for name, rois in ROIS.items():
    img=cv2.imread(f"images/{name}.png"); H,W=img.shape[:2]; g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    lines=[]; vis=img.copy()
    for (x0,x1,y0,y1) in rois:
        p=profile(g,(x0,x1,y0,y1)); per=count_layers(p)[2] or 20
        b=segment(p,pmin=max(8,int(0.62*per)),pmax=int(1.7*per))
        edges=[0]+list(b)+[y1-y0]
        for a,c in zip(edges[:-1],edges[1:]):
            if c-a<6: continue
            ya,yc=y0+a,y0+c
            lines.append(f"0 {(x0+x1)/2/W:.6f} {(ya+yc)/2/H:.6f} {(x1-x0)/W:.6f} {(yc-ya)/H:.6f}")
            cv2.rectangle(vis,(x0,ya),(x1,yc),(0,0,255),1)
    cv2.imwrite(f"{OUT}/images/{name}.png",img); cv2.imwrite(f"{OUT}/preview/{name}.png",vis)
    open(f"{OUT}/labels/{name}.txt","w").write("\n".join(lines)+"\n")
    print(name,len(lines),"pre-label boxes")
open(f"{OUT}/classes.txt","w").write("plank\n")
