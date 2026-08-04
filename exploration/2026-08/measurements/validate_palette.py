"""Python port of the dataviz skill's validate_palette.js (no node on this machine).

Same constants, same Machado-Oliveira-Fernandes (2009) severity-1.0 transforms, same OKLab
deltaE x100. Ported so the palette is computed rather than eyeballed, as the skill requires.
"""
import math, sys

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR, CVD_TARGET, CVD_FLOOR, NORMAL_FLOOR, CONTRAST_MIN = 0.10, 8.0, 6.0, 15.0, 3.0
SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}
MACHADO = {
 "protan": [[0.152286,1.052583,-0.204868],[0.114503,0.786281,0.099216],[-0.003882,-0.048116,1.051998]],
 "deutan": [[0.367322,0.860646,-0.227968],[0.280085,0.672501,0.047413],[-0.011820,0.042940,0.968881]],
 "tritan": [[1.255528,-0.076749,-0.178779],[-0.078411,0.930809,0.147602],[0.004733,0.691367,0.303900]]}

s2lin = lambda c: c/12.92 if c <= 0.04045 else ((c+0.055)/1.055)**2.4
def lin(h):
    h = h.lstrip("#"); return [s2lin(int(h[i:i+2],16)/255) for i in (0,2,4)]
def oklab_from_lin(rgb):
    r,g,b = rgb
    l = (0.4122214708*r+0.5363325363*g+0.0514459929*b)**(1/3)
    m = (0.2119034982*r+0.6806995451*g+0.1073969566*b)**(1/3)
    s = (0.0883024619*r+0.2817188376*g+0.6299787005*b)**(1/3)
    return (0.2104542553*l+0.7936177850*m-0.0040720468*s,
            1.9779984951*l-2.4285922050*m+0.4505937099*s,
            0.0259040371*l+0.7827717662*m-0.8086757660*s)
def oklch(h):
    L,a,b = oklab_from_lin(lin(h)); return L, math.hypot(a,b)
def simulate(h, kind):
    r,g,b = lin(h); M = MACHADO[kind]
    return [min(1,max(0,M[i][0]*r+M[i][1]*g+M[i][2]*b)) for i in range(3)]
def dE(h1,h2,kind=None):
    a = oklab_from_lin(simulate(h1,kind) if kind else lin(h1))
    b = oklab_from_lin(simulate(h2,kind) if kind else lin(h2))
    return 100*math.dist(a,b)
def rel_lum(h):
    r,g,b = lin(h); return 0.2126*r+0.7152*g+0.0722*b
def contrast(a,b):
    hi,lo = sorted((rel_lum(a),rel_lum(b)),reverse=True); return (hi+0.05)/(lo+0.05)

def validate(pal, mode="light", pairs="all"):
    surf = SURFACE[mode]; lo,hi = BAND[mode]; ok = True
    off = [(c,round(oklch(c)[0],3)) for c in pal if not lo <= oklch(c)[0] <= hi]
    ok &= not off
    print(f"  Lightness band      {'PASS' if not off else 'FAIL'}  " +
          (f"outside L {lo}-{hi}: {off}" if off else f"all {len(pal)} inside L {lo}-{hi}"))
    low = [(c,round(oklch(c)[1],3)) for c in pal if oklch(c)[1] < CHROMA_FLOOR]
    ok &= not low
    print(f"  Chroma floor        {'PASS' if not low else 'FAIL'}  " +
          (f"below floor: {low}" if low else f"all >= {CHROMA_FLOOR}"))
    n = len(pal)
    pl = ([(i,j) for i in range(n) for j in range(i+1,n)] if pairs=="all"
          else [(i,i+1) for i in range(n-1)])
    worst = min(((dE(pal[i],pal[j],k),k,pal[i],pal[j]) for k in ("protan","deutan") for i,j in pl))
    tri = min(dE(pal[i],pal[j],"tritan") for i,j in pl)
    st = "PASS" if worst[0] >= CVD_TARGET else "FLOOR" if worst[0] >= CVD_FLOOR else "FAIL"
    ok &= st != "FAIL"
    print(f"  CVD separation      {st:5s} worst {pairs} {worst[2]}<->{worst[3]} dE {worst[0]:.1f} "
          f"({worst[1]}) - tritan {tri:.1f}")
    nw = min(((dE(pal[i],pal[j]),pal[i],pal[j]) for i,j in pl))
    ok &= nw[0] >= NORMAL_FLOOR
    print(f"  Normal-vision floor {'PASS' if nw[0]>=NORMAL_FLOOR else 'FAIL'}  worst {nw[1]}<->{nw[2]} "
          f"dE {nw[0]:.1f} (floor {NORMAL_FLOOR:.0f})")
    lc = [(c,round(contrast(c,surf),2)) for c in pal if contrast(c,surf) < CONTRAST_MIN]
    print(f"  Contrast vs surface {'PASS' if not lc else 'RELIEF'} " +
          (f" below 3:1 -> needs visible labels/table: {lc}" if lc else f" all >= {CONTRAST_MIN}:1"))
    return ok

if __name__ == "__main__":
    pal = [c.strip() for c in sys.argv[1].split(",")]
    mode = sys.argv[2] if len(sys.argv) > 2 else "light"
    print(f"mode={mode} surface={SURFACE[mode]} palette={pal}")
    print("OK" if validate(pal, mode) else "NOT OK")
