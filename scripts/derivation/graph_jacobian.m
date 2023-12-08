clear all
close all
syms xr1 yr1 theta1 xp yp zp zx zy zz xr2 yr2 theta2 ztheta real
%% feature error
c1=cos(theta1);
s1=sin(theta1);
x=[xp;yp;zp;1];
z=[zx;zy;zz];
T1=[c1 -s1 0 xr1
    s1 c1 0 yr1
    0 0 1 0
    0 0 0 1];

e=simplify(T1^(-1)*x);
e=simplify(z-e(1:3))

J1f=simplify([diff(e, xr1)  diff(e, yr1) diff(e, theta1)])

J2f=simplify([diff(e, xp)  diff(e, yp) diff(e, zp)])

%% pose error 
z2=[zx; zy; ztheta];
cz=cos(ztheta);
sz=sin(ztheta);
Z=[cz -sz 0 zx
    sz cz 0 zy
    0 0 1 0
    0 0 0 1];
c2=cos(theta2);
s2=sin(theta2);

T2=[c2 -s2  0 xr2
    s2 c2  0 yr2
    0 0  1 0
    0 0 0 1];
E2=simplify(Z^(-1)*(T1^(-1)*T2))
e2=[E2(1,4); E2(2,4); atan2(E2(2,1), E2(1,1)) ]

J1p=simplify([diff(e2, xr1) diff(e2, yr1) diff(e2, theta1)])
J2p=simplify([diff(e2, xr2) diff(e2, yr2) diff(e2, theta2)])

