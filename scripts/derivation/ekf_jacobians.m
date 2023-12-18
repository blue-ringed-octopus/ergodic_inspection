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


%% landmark orientation
clear all
close all
clc; 
syms thetar thetatag wx wy wz real
Rr=[cos(thetar) sin(thetar) 0
    -sin(thetar) cos(thetar) 0
    0 0 1];
R_tag=[cos(thetatag) sin(thetatag) 0 
    -sin(thetatag) cos(thetatag) 0
    0 0 1];

R_tag_c=Exp([wx;wy;wz]);


R_bar=Rr'*R_tag;
T=[0 0 1
    0 -1 0
    1 0 0]
Log(T)
r=simplify(Log(R_bar));
% 
% e=simplify(Log(R_bar'*R_tag_c))

% J=[diff(r, thetar), diff(r, thetatag)]
% 
test=simplify(Jr_inv(r))
%%
w1=rand(3,1);
w2=rand(3,1);


R1=Exp(w1);
R2=Exp(w2);

dR=R1'*R2;

dw=w2-w1
Log(Exp(dw))
dw2=Log(dR)
function R=Exp(w)
theta=norm(w);
u=w/theta;
R=eye(3)+hat(u)*sin(theta)+(1-cos(theta))*hat(u)^2;
end
function w=Log(R)
theta=acos((trace(R)-1)/2);
w=theta*vee(R-R')/(2*sin(theta));
end
function w=vee(W)
w=[W(3,2); W(1,3); W(2,1)];
end

function R=hat(w)
R=[0 -w(3) w(2)
    w(3) 0 -w(1)
    -w(2) w(1) 0];
end

function J=Jr(w)
t=norm(w);
J=eye(3)-(eye(3)-cos(t))/t^2*hat(w)+(t-sin(t))/t^3*hat(w)*hat(w);
end

function J=Jr_inv(w)
t=norm(w);
J=eye(3)+1/2*hat(w)+(1/t^2-(1+cos(t))/(2*t*sin(t)))*hat(w)*hat(w);
end

function J=Jl_inv(w)
t=norm(w);
J=eye(3)-1/2*hat(w)+(1/t^2-(1+cos(t))/(2*t*sin(t)))*hat(w)*hat(w);
end
