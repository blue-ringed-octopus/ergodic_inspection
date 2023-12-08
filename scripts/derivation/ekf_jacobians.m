syms x y theta  dx dy dtheta real

c1=cos(theta)
s1=sin(theta)
c2=cos(dtheta)
s2=sin(dtheta)

X=[c1 -s1 x
    s1 c1 y
    0 0 1]

dX=[c2 -s2 dx
    s2 c2 dy
    0 0 1];

X2=X*dX

x2=simplify([X2(1,3), X2(2,3), atan2(X2(2,1), X2(1,1))])'

fu=simplify([diff(x2, dx), diff(x2,dy), diff(x2, dtheta)])

fx=simplify([diff(x2, x), diff(x2,y), diff(x2, theta)])

%%
clear all
syms fx fy cx cy rx ry rz dx dy dz x y real  

K=[fx 0 cx;
    0 fy cy
    0 0 1];

c=cos(rz)
s=sin(rz)
M=[c -s dx
    s c dy
    0 0 1]
V=s/rz*eye(2)+(1-c)/rz*[0 -1
                        1 0 ]
                    
V_inv=simplify(inv(V));

rho=simplify(V_inv*[dx;dy])
t=[rho; rz]


%%
syms dtheta

dx=0
dy=1 

dtheta=0.000001
u=[dx;dy;dtheta];
s=sin(dtheta);
c=cos(dtheta);
V=s/dtheta*eye(2)+(1-c)/dtheta*[0 -1
                        1 0 ]
                    
R=[c -s
    s c];

M=([R V*u(1:2)
    0 0 1])
