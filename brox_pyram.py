import numpy as np
from numpy.core.fromnumeric import reshape
from math import floor


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
import  scipy.misc 
#####################################################################################
def resize_function_CtoF(ns,Im,tab):
    '''(n,m): shape of the image Im & ns the pyramid level, reshape from a coarsest to  a finer level  '''
    (new_N,new_M)=tab[ns]
    Ims=scipy.misc.imresize(Im,(new_N,new_M),interp='bilinear')
    return Ims

def resize_function_FtoC(u,tab,ns):
    '''(n,m): shape of the dispacement u & ns the pyramid level, reshape from a finer level to  the next coarsest one '''
    '''(n,m)=u.shape
    new_N=round(n/factor)
    new_M=round(m/factor)'''
    (new_N,new_M)=tab[ns-1]
    new_u=scipy.misc.imresize(u,(new_N,new_M),interp='bilinear')
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
def brox_method(Im1,Im2,u,v,gamma,alpha,inner_it,outer_it,a,eps,w,iter_SOR,tol,nscales,nu,gaussian_sigma):
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
        if(ns==0):
            Im1s=Im1; Im2s=Im2
        print('Im shape',Im1s.shape)
        [u,v]=brox.brox_opt_flow(Im1s,Im2s,u,v,gamma,alpha,inner_it,outer_it,a,eps,w,iter_SOR,tol)
        if(ns!=0):
            u=resize_function_FtoC(u,tab,ns)
            v=resize_function_FtoC(v,tab,ns)

            u=1/nu*u
            v=1/nu*v
    

    return[u,v]




########################################################################################
'''u=np.random.rand(5,3); 
factor=0.5; ns=2
#Ims=resize_function_CtoF(ns,Im,factor)
new_u=resize_function_FtoC(u,factor)
print(new_u.shape)'''

Im1=np.random.rand(20,30);
Im2=np.random.rand(20,30);
u=np.random.rand(20,30);
v=np.random.rand(20,30);
'''u=np.zeros((40,50))
v=np.zeros((40,50))'''
gamma=0.5
alpha=0.2
inner_it=10
outer_it=10
a=0.5
eps=0.001
w=0.5
iter_SOR=30
tol=0.01
nscales=3
nu=0.5
gaussian_sigma=0.9
[u,v]=brox_method(Im1,Im2,u,v,gamma,alpha,inner_it,outer_it,a,eps,w,iter_SOR,tol,nscales,nu,gaussian_sigma)