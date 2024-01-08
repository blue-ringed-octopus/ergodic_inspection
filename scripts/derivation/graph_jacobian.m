clear all
close all
syms xr1 yr1 tr1 xl yl zl tl xr2 yr2 tr2 zx zy zt real
%% point feature error
c1=cos(tr1);
s1=sin(tr1);
x=[xl;yl;zl;1];
T1=[c1 -s1 0 xr1
    s1 c1 0 yr1
    0 0 1 0
    0 0 0 1];

e=simplify(T1^(-1)*x);
e=simplify(-e(1:3))

J1f=simplify([diff(e, xr1)  diff(e, yr1) diff(e, tr1)])

J2f=simplify([diff(e, xl)  diff(e, yl) diff(e, zl)])

%% feature pose error
cr=cos(tr1)
sr=sin(tr1)
cl=cos(tl)
sl=sin(tl)

Tr=[cr -sr 0 xr1
    sr cr 0 yr1
    0 0 1 0
    0 0 0 1]

Tl=[cl -sl 0 xl
    sl cl 0 yl
    0 0 1 zl
    0 0 0 1]

Z=simplify(Tr^(-1)*Tl)
z=simplify([Z(1,4), Z(2,4), Z(3,4), atan2(Z(2,1), Z(1,1))]')
e=-z
Jr=simplify([diff(e, xr1),diff(z, yr1), diff(e, tr1)])

Jl=simplify([diff(e, xl), diff(z, yl), diff(e, zl), diff(e, tl) ])

%% pose error 
z2=[zx; zy; zt];
cz=cos(zt);
sz=sin(zt);
Z=[cz -sz 0 zx
    sz cz 0 zy
    0 0 1 0
    0 0 0 1];
c2=cos(tr2);
s2=sin(tr2);

T2=[c2 -s2  0 xr2
    s2 c2  0 yr2
    0 0  1 0
    0 0 0 1];
E2=simplify(Z^(-1)*(T1^(-1)*T2))
e2=[E2(1,4); E2(2,4); atan2(E2(2,1), E2(1,1)) ]

J1p=simplify([diff(e2, xr1) diff(e2, yr1) diff(e2, tr1)])
J2p=simplify([diff(e2, xr2) diff(e2, yr2) diff(e2, tr2)])

