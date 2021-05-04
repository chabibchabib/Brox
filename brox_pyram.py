import numpy as np
from numpy.core.fromnumeric import reshape
from math import floor
from ssl import DefaultVerifyPaths
import matplotlib.pyplot as plt

import cv2
from math import ceil,floor
from scipy.ndimage.filters import convolve as filter2
from scipy.sparse import spdiags
from scipy.signal import medfilt
from scipy.ndimage import median_filter,gaussian_filter
import scipy.sparse as sparse
import math
from scipy.ndimage import map_coordinates
import brox_mod as brox 
#import  scipy.misc 
from skimage.transform import resize
#####################################################################################
def resize_function_CtoF(ns,Im,tab):
    '''(n,m): shape of the image Im & ns the pyramid level, reshape from a coarsest to  a finer level  '''
    (new_N,new_M)=tab[ns]
    #Ims=scipy.misc.imresize(Im,(new_N,new_M),interp='bilinear')
    Ims=resize(Im,(new_N,new_M))
    return Ims

def resize_function_FtoC(u,tab,ns):
    '''(n,m): shape of the dispacement u & ns the pyramid level, reshape from a finer level to  the next coarsest one '''

    (new_N,new_M)=tab[ns-1]
    #new_u=scipy.misc.imresize(u,(new_N,new_M),interp='bilinear')
    new_u=resize(u,(new_N,new_M))
    return new_u

def sizes(nscales,Im,factor):
    tab=[]
    (n,m)=Im.shape
    tab.append((n,m))
    for ns in range(1,nscales):
        new_N=round(n*factor**ns)
        new_M=round(m*factor**ns)
        tab.append((new_N,new_M))
    return tab
#####################################################################################
def brox_method(Im1,Im2,u,v,gamma,alpha,inner_it,outer_it,a,eps,w,iter_SOR,tol,nscales,nu,gaussian_sigma0):
    gaussian_sigma=gaussian_sigma0*math.sqrt(1/nu**2-1)
    Im1=gaussian_filter(Im1,gaussian_sigma)
    Im2=gaussian_filter(Im2,gaussian_sigma)
    tab=sizes(nscales,Im1,nu)
    u=resize_function_CtoF(nscales-1,u,tab)
    v=resize_function_CtoF(nscales-1,v,tab)
    print('u shape',u.shape)

    for ns in range(nscales-1,-1,-1):
        print('level:',ns)
        if (ns!=0):
            Im1s=resize_function_CtoF(ns,Im1,tab); Im2s=resize_function_CtoF(ns,Im2,tab)
            Im1s=gaussian_filter(Im1s,gaussian_sigma)
            Im2s=gaussian_filter(Im2s,gaussian_sigma)
            Im1s=np.array(Im1s,np.float32)
            Im2s=np.array(Im2s,np.float32)
        if(ns==0):
            Im1s=Im1; Im2s=Im2
        print('Im shape',Im1s.shape)
        [u,v]=brox.brox_opt_flow(Im1s,Im2s,u,v,gamma,alpha,inner_it,outer_it,a,eps,w,iter_SOR,tol)
        if(ns!=0):
            u=resize_function_FtoC(u,tab,ns)
            v=resize_function_FtoC(v,tab,ns)
            '''u=median_filter(u,(5,5))
            v=median_filter(v,(5,5))'''
            u=1/nu*u
            v=1/nu*v
        print(u)
    return[u,v]
########################################################################################
#Im1=np.random.rand(20,30);
#Im2=np.random.rand(20,30);
#u=np.random.rand(20,30);
#v=np.random.rand(20,30);
Im1=cv2.imread('Im1.png',0)
Im2=cv2.imread('Im2.png',0)
Im1=np.array(Im1,dtype=np.float32)
Im2=np.array(Im2,dtype=np.float32)
u=np.zeros((Im1.shape[0],Im1.shape[1]))
v=u=np.zeros((Im1.shape[0],Im1.shape[1]))

'''gamma=20
alpha=15
inner_it=20
outer_it=4
a=0.5
eps=0.0001
w=0.5
iter_SOR=500
tol=0.0001
nscales=5
nu=0.5
gaussian_sigma=0.6'''
gamma=30
alpha=2
inner_it=20
outer_it=2
a=0.5
eps=0.0001
w=1
iter_SOR=700
tol=0.0001
nscales=6
nu=0.5
gaussian_sigma=0.6
[u,v]=brox_method(Im1,Im2,u,v,gamma,alpha,inner_it,outer_it,a,eps,w,iter_SOR,tol,nscales,nu,gaussian_sigma)

N,M=Im1.shape
y=np.linspace(0,N-1,N)
x=np.linspace(0,M-1,M)
x,y=np.meshgrid(x,y)
x2=x+u; y2=y+v
x2=np.array(x2,dtype=np.float32)
y2=np.array(y2,dtype=np.float32)
I=cv2.remap(np.array(Im1,dtype=np.float32),x2,y2,cv2.INTER_LINEAR)
norme=np.linalg.norm(I-Im2)/np.linalg.norm(Im2)
print(norme)
cv2.imwrite('I_2.png',I)
