import numpy as np
from numpy.core.fromnumeric import reshape
import cv2
from math import ceil,floor
from scipy.ndimage.filters import convolve as filter2
from scipy.sparse import spdiags
from scipy.signal import medfilt
from scipy.ndimage import median_filter,gaussian_filter
import scipy.sparse as sparse
import math
from scipy.ndimage import map_coordinates
##############################################
def psi_data(I1,I2,I2x,I2y,du,dv,eps,a):
    ''' Computing psi_data'''
    dI=I2-I1+ I2x*du+I2y*dv
    dI=dI**2
    #psip = 1/ math.sqrt(dI + eps**2);
    psip = (dI + eps**2)**(a-1)
    return psip
###########################################
def psi_gradient(I1x,I1y,I2x,I2y,I2xx,I2yy, I2xy,du, dv,eps,a):
    ''' Compute psi_gradient'''

	#psip = 1. / math.sqrt(dI2 + eps **2);
    dIx = I2x- I1x+I2xx*du+I2xy*dv
    dIy = I2y- I1y+ I2xy * du + I2yy * dv
    dI2 = dIx * dIx + dIy * dIy
    psip = (dI2 + eps**2)**(a-1)
    return psip

###########################################
def psi_smooth(ux,uy,vx,vy,a,eps):
    ''' Compute psi_smooth'''
    du=ux**2+ uy**2
    dv= vx**2 + vy**2
    d2= du + dv
    #psi= 1 / math.sqrt(d2 + eps**2)
    psip = (d2 + eps**2)**(a-1)
    return psip
###########################################
def gradient(Im):
    '''compute gradient of an image '''

    (Iy,Ix)=np.gradient(Im)
    return [Ix,Iy]
###########################################
def second_deriv(Im):
    '''compute second derivatives of an image  (ie Ixx,Ixy or Iyy) '''
    Iyy=np.gradient(Im,axis=0)
    Iyy=np.gradient(Iyy,axis=0)
    Ixx=np.gradient(Im,axis=1)
    Ixy=np.gradient(Ixx,axis=0)
    Ixx=np.gradient(Ixx,axis=1)
    return [Ixx,Iyy,Ixy]
###############################################
def image_interpolation(Im,u,v):
    '''warping an image using bicubic interpolation with reflecting boundary conditions (with the help of map_coordinates function) '''
    (n,m)=Im.shape
    y=np.linspace(0,n-1,n)
    x=np.linspace(0,m-1,m)
    x,y=np.meshgrid(x,y)
    
    x=x+u; 
    y=y+v
    '''print('x')
    print(x)
    print('y')
    print(y)'''
    x=np.reshape(x,(1,n*m),'F')
    y=np.reshape(y,(1,n*m),'F')
   

    Iw=map_coordinates(Im,[y,x],order=3,mode='reflect')

    Iw=np.reshape(Iw,(n,m),'F')
    '''print('A_warp')
    print(Iw)'''
    return Iw
################################################
def psi_divergence(psi):
    (n,m)=psi.shape
    psi1=np.empty((n,m))
    psi2=np.empty((n,m))
    psi3=np.empty((n,m))
    psi4=np.empty((n,m))
    #Center nodes 
    psi1[1:n-1,1:m-1]=psi[2:n,1:m-1]+psi[1:n-1,1:m-1]
    psi2[1:n-1,1:m-1]=psi[0:n-2,1:m-1]+psi[1:n-1,1:m-1]
    psi3[1:n-1,1:m-1]=psi[1:n-1,2:m]+psi[1:n-1,1:m-1]
    psi4[1:n-1,1:m-1]=psi[1:n-1,0:m-2]+psi[1:n-1,1:m-1]
    # First & last row
    psi1[0,1:m-1]=psi[1,1:m-1]+psi[0,1:m-1]
    psi2[0,1:m-1]=0
    psi3[0,1:m-1]=psi[0,2:m]+psi[0-1,1:m-1]
    psi4[0,1:m-1]=psi[0,0:m-2]+psi[0-1,1:m-1]

    psi1[n-1,1:m-1]=0
    psi2[n-1,1:m-1]=psi[n-2,1:m-1]+psi[n-1,1:m-1]
    psi3[n-1,1:m-1]=psi[n-1,2:m]+psi[n-1,1:m-1]
    psi4[n-1,1:m-1]=psi[n-1,0:m-2]+psi[n-1,1:m-1]
    #First and last columns 
    psi1[1:n-1,0]=psi[2:n,0]+psi[1:n-1,0]
    psi2[1:n-1,0]=psi[0:n-2,0]+psi[1:n-1,0]
    psi3[1:n-1,0]=psi[1:n-1,1]+psi[1:n-1,0]
    psi4[1:n-1,0]=0

    psi1[1:n-1,m-1]=psi[2:n,m-1]+psi[1:n-1,m-1]
    psi2[1:n-1,m-1]=psi[0:n-2,0]+psi[1:n-1,0]
    psi3[1:n-1,m-1]=0
    psi4[1:n-1,m-1]=psi[1:n-1,m-2]+psi[1:n-1,m-1]
    #Corners
    psi1[0,0]=psi[0,1]+psi[0,0]
    psi2[0,0]=0; psi4[0,0]=0
    psi3[0,0]=psi[1,0]+psi[0,0]

    psi1[0,m-1]=psi[1,m-1]+psi[0,m-1]
    psi2[0,m-1]=0
    psi3[0,m-1]=0
    psi4[0,m-1]=psi[0,m-2]+psi[0,m-1]

    psi1[n-1,0]=0
    psi2[n-1,0]=psi[n-2,0]+psi[n-1,0]
    psi3[n-1,0]=psi[n-1,1]+psi[n-1,0]
    psi4[n-1,0]=0

    psi1[n-1,m-1]=0
    psi2[n-1,m-1]=psi[n-2,m-1]+psi[n-1,m-1]
    psi3[n-1,m-1]=0
    psi4[n-1,m-1]=psi[n-1,m-1]+psi[n-1,m-2]

    psi1=0.5*psi1
    psi2=0.5*psi2
    psi3=0.5*psi3
    psi4=0.5*psi4
    return[psi1,psi2,psi3,psi4]
################################################
def divergence_u(u,psi1,psi2,psi3,psi4):
    (n,m)=u.shape
    div_u= np.empty((n,m),dtype=np.float32); #div_v= np.empty((n,m))
    div_u[1:n-1,1:m-1]=psi1[1:n-1,1:m-1]*(u[2:n,1:m-1]-u[1:n-1,1:m-1]) +psi2[1:n-1,1:m-1]*(u[0:n-2,1:m-1]-u[1:n-1,1:m-1])+psi3[1:n-1,1:m-1]*(u[1:n-1,2:m]-u[1:n-1,1:m-1])
    +psi4[1:n-1,1:m-1]*(u[1:n-1,0:m-2]-u[1:n-1,1:m-1])

    div_u[0,1:m-1]=psi1[0,1:m-1]*(u[1,1:m-1]-u[0,1:m-1]) +psi3[0,1:m-1]*(u[0,2:m]-u[0,1:m-1])+psi4[0,1:m-1]*(u[0,0:m-2]-u[0,1:m-1]) # first row
    div_u[n-1,1:m-1]=psi2[n-1,1:m-1]*(u[n-2,1:m-1]-u[n-1,1:m-1]) +psi3[n-1,1:m-1]*(u[n-1,2:m]-u[n-1,1:m-1])+psi4[n-1,1:m-1]*(u[n-1,0:m-2]-u[n-1,1:m-1]) # last row
    div_u[1:n-1,0]=psi1[1:n-1,0]*(u[2:n,0]-u[1:n-1,0])+ psi2[1:n-1,0]*(u[0:n-2,0]-u[1:n-1,0]) +psi3[1:n-1,0]*(u[1:n-1,1]-u[1:n-1,0])# first col
    div_u[1:n-1,m-1]=psi1[1:n-1,m-1]*(u[2:n,m-1]-u[1:n-1,m-1]) +psi2[1:n-1,m-1]*(u[0:n-2,m-1]-u[1:n-1,m-1])+psi4[1:n-1,m-1]*(u[1:n-1,m-2]-u[1:n-1,m-1])# last col

    div_u[0,0]=psi1[0,0]*(u[1,0]-u[0,0])+psi3[0,0]*(u[0,1]-u[0,0]) #(0,0)
    div_u[0,m-1]=psi1[0,m-1]*(u[1,m-1]-u[0,m-1])+psi4[0,m-1]*(u[0,m-2]-u[0,m-1]) #(0,m-1)
    div_u[n-1,0]=psi2[n-1,0]*(u[n-2,m-1]-u[n-1,0])+psi3[n-1,0]*(u[n-1,1]-u[n-1,0]) ##(n-1,0)
    div_u[n-1,m-1]=psi2[n-1,m-1]*(u[n-2,m-1]-u[n-1,m-1])+psi4[n-1,m-1]*(u[n-1,m-2]-u[n-1,m-1]) ##(n-1,m-1)
    return div_u

################################################
def div_du(psi1,psi2,psi3,psi4,du):
    (n,m)=du.shape
    div_du= np.empty((n,m),dtype=np.float32)
    div_du[1:n-1,1:m-1]=psi1[1:n-1,1:m-1]*du[2:n,1:m-1]+psi2[1:n-1,1:m-1]*du[0:n-2,1:m-1]+psi3[1:n-1,1:m-1]*du[1:n-1,2:m]+psi4[1:n-1,1:m-1]*du[1:n-1,0:m-2]

    div_du[0,1:m-1]=psi1[0,1:m-1]*du[1,1:m-1]+psi3[0,1:m-1]*du[0,2:m]+psi4[0,1:m-1]*du[0,0:m-2]
    div_du[n-1,1:m-1]=psi2[n-1,1:m-1]*du[n-2,1:m-1]+psi3[n-1,1:m-1]*du[n-1,2:m]+psi4[n-1,1:m-1]*du[n-1,0:m-2]

    div_du[1:n-1,0]=psi1[1:n-1,0]*du[2:n,0]+psi2[1:n-1,0]*du[0:n-2,0]+psi3[1:n-1,0]*du[1:n-1,1]
    div_du[1:n-1,m-1]=psi1[1:n-1,m-1]*du[2:n,m-1]+psi2[1:n-1,m-1]*du[0:n-2,m-1]+psi4[1:n-1,m-1]*du[1:n-1,m-2]

    div_du[0,0]=psi1[0,0]*du[1,0]+psi3[0,0]*du[0,1]
    div_du[0,m-1]=psi1[0,m-1]*du[1,m-1]+psi4[0,m-1]*du[0,m-2]
    div_du[n-1,0]=psi2[n-1,0]*du[n-2,0]+psi3[n-1,0]*du[n-1,1]
    div_du[n-1,m-1]=psi2[n-1,m-1]*du[n-2,m-1]+psi4[n-1,m-1]*du[n-1,m-2]
    return div_du

################################################

def brox_opt_flow(Im1,Im2,u,v,gamma,alpha,inner_it,outer_it,a,eps,w,iter_SOR,tol):
    (n,m)=Im1.shape
    [I1x,I1y]=gradient(Im1)
    [I2x,I2y]=gradient(Im2)

    [I2xx,I2yy,I2xy]=second_deriv(Im2)
    for i in range(outer_it):
        I2w=image_interpolation(Im2,u,v)
        I2xw=image_interpolation(I2x,u,v)
        I2yw=image_interpolation(I2y,u,v)
        I2xxw=image_interpolation(I2xx,u,v)
        I2yyw=image_interpolation(I2yy,u,v)
        I2xyw=image_interpolation(I2xy,u,v)

        [ux,uy]=gradient(u)
        [vx,vy]=gradient(v)
        psis=psi_smooth(ux,uy,vx,vy,a,eps)
        [psi1,psi2,psi3,psi4]=psi_divergence(psis)
        div_u=divergence_u(u,psi1,psi2,psi3,psi4)
        div_v=divergence_u(v,psi1,psi2,psi3,psi4)
        div_d=alpha*(psi1+ psi2 + psi3 + psi4)

        du=np.zeros(u.shape); dv=np.zeros(u.shape)

        for j in range(inner_it):
            psid=psi_data(Im1,I2w,I2xw,I2yw,du,dv,eps,a)
            psig=psi_gradient(I1x,I1y,I2xw,I2yw,I2xxw,I2yyw, I2xyw,du, dv,eps,a)
            dif = I2w-Im1
            dif_x=I2xw-I1x;dif_y=I2yw-I1y

            Au=-psid*dif*I2xw; Au=Au+alpha*div_u; Au=Au-gamma*psig*(dif_x*I2xxw+dif_y*I2xyw)
            Av=-psid*dif*I2yw; Av=Av+alpha*div_v; Av=Av-gamma*psig*(dif_x*I2xyw+dif_y*I2yyw)

            Du=psid**I2xw**2; Du=Du+gamma*psig*(I2xxw**2+I2xyw**2); Du=Du+alpha*div_d
            Dv=psid**I2yw**2; Dv=Dv+gamma*psig*(I2yyw**2+I2xyw**2); Dv=Dv+alpha*div_d
            D=psid*I2xw*I2yw+gamma*psig*(I2xxw+I2yyw)*I2xyw

            error=1000; sor_it=0
            while((error >tol) or sor_it<iter_SOR ):
                error=0; sor_it=sor_it+1

                div_duu=div_du(psi1,psi2,psi3,psi4,du)
                div_dv=div_du(psi1,psi2,psi3,psi4,dv)
                du0=du; dv0=dv
                du= (1-w) * du + w * (Au - D * dv + alpha * div_duu) / Du
                dv= (1-w) * dv + w * (Av - D * du + alpha * div_dv) / Dv
                error=np.linalg.norm((du-du0))+np.linalg.norm((dv-dv0))
                error=math.sqrt(error/n*m)
            
            u=u+du; v=v+dv 
        
    return[u,v]




        





    
################################################
#A=np.floor(10*np.random.rand(4,4)) 
'''A=[[6., 3., 4. ,8.,6.]
 ,[6. ,5., 3. ,1.,0.]
 ,[9. ,1. ,9., 5.,1.]
 ,[5., 8. ,4., 6.,7.]]
A=np.array(A)
print('A')
print(A)
[psi1,psi2,psi3,psi4]=psi_divergence(A)
u=np.ones(A.shape); v=2*np.ones(A.shape)
#u=np.random.rand(4,5)
#div_u=divergence_u(u,psi1,psi2,psi3,psi4)

[u,v]=brox_opt_flow(A,2*A,u,v,1,2,3,4,0.5,0.001,0.5,5,0.001)

print(u)
print(v)'''