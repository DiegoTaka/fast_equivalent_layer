from __future__ import division
import numpy as np
from scipy.sparse import diags
from scipy.linalg import toeplitz
from numpy.linalg import inv

#: The default thermal diffusivity in :math:`m^2/s`
THERMAL_DIFFUSIVITY = 0.000001

#: The default thermal diffusivity but in :math:`m^2/year`
THERMAL_DIFFUSIVITY_YEAR = 31.5576

#: Conversion factor from SI units to Eotvos: :math:`1/s^2 = 10^9\ Eotvos`
SI2EOTVOS = 1000000000.0

#: Conversion factor from SI units to mGal: :math:`1\ m/s^2 = 10^5\ mGal`
SI2MGAL = 100000.0

#: The gravitational constant in :math:`m^3 kg^{-1} s^{-1}`
G = 0.00000000006673

#: Proportionality constant used in the magnetic method in henry/m (SI)
CM = 10. ** (-7)

#: Conversion factor from tesla to nanotesla
T2NT = 10. ** (9)

#: The mean earth radius in meters
MEAN_EARTH_RADIUS = 6378137.0

#: Permeability of free space in :math:`N A^{-2}`
PERM_FREE_SPACE = 4 * \
    3.141592653589793115997963468544185161590576171875 * (10 ** -7)

def fast_eq(x,y,z,h,shape,data,itmax):
    '''
    Calculates the estimate physical property distribution of
    an equivalent layer that repreduces a gravity disturbance
    data through an iterative method [1].

    [1] SIQUEIRA, F. C., OLIVEIRA JR, V. C., BARBOSA, V. C., 2017,
    Fast iterative equivalent-layer technique for gravity data
    processing: A method grounded on excess mass constraint",
    Geophysics, v. 82, n. 4, pp. G57-G69.

    input
    x, y: numpy array - the x, y coordinates
    of the grid and equivalent layer points.
    z: numpy array - the height of observation points.
    h: numpy array - the depth of the equivalent layer.
    shape: tuple - grid size.
    data: numpy array - the gravity disturbance.
    potential field at the grid points.

    output
    m_new: numpy array - final equivalent
    layer property estimative.
    gzp: numpy array - the predicted data.
    '''
    assert x.size == y.size == z.size == h.size == data.size, 'x, y,\
    z, h and data must have the same number of elements'
    #assert h.all() > z.all(), 'The equivalent layer must be beneath\
    #the observation points'

    #Diagonal matrix calculation
    N,diagonal_A = diagonal(x,y,shape)

    #Initial estimative
    rho0 = i_rho(data,diagonal_A)
    
    #Complete sensibility matrix
    A = sensibility_matrix(x,y,z,h,N)

    #Fast Equivalent layer loop
    m_new = fast_loop(data,A,diagonal_A,rho0,itmax)

    #Final predicted data
    gzp = A.dot(m_new)
    
    A = np.zeros((N, N), dtype=np.float)

    return m_new, gzp
    
def fast_eq_bccb(x,y,z,h,shape,data,itmax):
    '''
    Calculates the estimate physical property distribution of
    an equivalent layer that repreduces a gravity disturbance
    data through an iterative method [1]. 
    This implementation uses a fast way to calculate the forward
    problem at each iteration taking advantage of the BTTB
    (Block-Toeplitz Toreplitz-Block) structures.

    [1] SIQUEIRA, F. C., OLIVEIRA JR, V. C., BARBOSA, V. C., 2017,
    Fast iterative equivalent-layer technique for gravity data
    processing: A method grounded on excess mass constraint",
    Geophysics, v. 82, n. 4, pp. G57-G69.

    input
    x, y: numpy array - the x, y coordinates
    of the grid and equivalent layer points.
    z: numpy array - the height of observation points.
    h: numpy array - the depth of the equivalent layer.
    shape: tuple - grid size.
    data: numpy array - the gravity disturbance.
    potential field at the grid points.

    output
    m_new: numpy array - final equivalent
    layer property estimative.
    gzp: numpy array - the predicted data.
    '''
    assert x.size == y.size == z.size == h.size == data.size, 'x, y,\
    z, h and data must have the same number of elements'
    #assert h.all() > z.all(), 'The equivalent layer must be beneath\
	#the observation points'
    
    #Diagonal matrix calculation
    N,diagonal_A = diagonal(x,y,shape)

    #Initial estimative
    rho0 = i_rho(data,diagonal_A)
    
    #Create first line of sensibility matrix
    BTTB = bttb(x,y,z,h)
    
    #Calculates the eigenvalues of BCCB matrix
    cev = bccb(shape,N,BTTB)

    #Fast Equivalent layer loop
    m_new = fast_loop_bccb(cev,shape,N,data,diagonal_A,rho0,itmax)

    #Final predicted data
    gzp = fast_forward_bccb(shape,N,m_new,cev)

    return m_new, gzp

def diagonal(x,y,shape):
    '''
    Calculates a NxN diagonal matrix given by
    the area of x and y data spacing.

    input
    x, y: numpy array - the x, y coordinates of
    the grid and equivalent layer points.
    shape: tuple - grid size.

    output
    N: scalar - number of observation points.
    diagonal_A: escalar - NxN diagonal matrix
    given by the area of x nd y data spacing.
    '''
    N = shape[0]*shape[1]
    assert N == x.size, 'N and x must have the same number\
    of elements'
    delta_s = ((np.max(x)-np.min(x))/(shape[0]-1))*\
    ((np.max(y)-np.min(y))/(shape[1]-1))
    diagonal_A = G*SI2MGAL*2.*np.pi/(delta_s)
    return N,diagonal_A

def i_rho(data,diagonal_A):
    '''
    Calculates the initial equivalent layer
    property estimative.

    input
    data: numpy array - the gravity disturbance.
    diagonal_A: escalar - NxN diagonal matrix
    given by the area of x nd y data spacing.

    output
    rho0: numpy array - initial equivalent
    layer property estimative.
    '''
    rho0 = data/diagonal_A
    return rho0

def bttb(x,y,z,h):
    '''
    Calculates the first line of sensbility matrix.

    input
    x, y: numpy array - the x, y coordinates
    of the grid and equivalent layer points.
    z: numpy array - the height of observation points.
    h: numpy array - the depth of the equivalent layer.

    output
    W_bt: numpy array - first line os sensibility matrix.
    '''
    a = (x-x[0])
    b = (y-y[0])
    c = (h-z[0])
    W_bt = (G*SI2MGAL*c)/((a*a+b*b+c*c)**(1.5))
    return W_bt

def bccb(shape,N,BTTB):
    '''
    Calculates the eigenvalues of the BCCB matrix.

    input
    shape: tuple - grid size.
    N: scalar - number of observation points.
    BTTB: numpy array - first line os sensibility matrix.

    output
    cev: numpy array - eigenvalues of the BCCB matrix.
    '''
    cev = np.zeros(4*N, dtype='complex128')
    k = 2*shape[0]-1
    for i in range (shape[0]):
        block = BTTB[shape[1]*(i):shape[1]*(i+1)]
        rev = block[::-1]
        cev[shape[1]*(2*i):shape[1]*(2*i+2)] = np.concatenate((block,0,rev[:-1]), axis=None)
        if i > 0:
            cev[shape[1]*(2*k):shape[1]*(2*k+2)] = cev[shape[1]*(2*i):shape[1]*(2*i+2)]
            k -= 1
    cev = cev.reshape(2*shape[0],2*shape[1]).T
    return np.fft.fft2(cev)

def sensibility_matrix(x,y,z,h,N):
    '''
    Calculates a full NxN matrix given by
    the first derivative of the function
    1/r.

    input
    x, y: numpy array - the x, y coordinates of
    the grid and equivalent layer points.
	z: numpy array - the height of observation points.
    h: numpy array - the depth of the equivalent layer.
    N: scalar - number of observation points.

    output
    A: matrix - full NxN matrix given by
    the first derivative of the function
    1/r.
    '''
    A = np.zeros((N, N), dtype=np.float)

    for i in range (N):
        a = (x-x[i])
        b = (y-y[i])
        c = (h-z[i])
        A[i] = c/((a*a+b*b+c*c)**(1.5))
    A = A*G*SI2MGAL
    return A

def fast_loop(data,A,diagonal_A,rho0,itmax):
    '''
    Solves the linear inversion through a iterative method.

    input
    data: numpy array - the gravity disturbance.
	A: matrix - full NxN matrix given by
    the first derivative of the function
    1/r.
    diagonal_A: escalar - NxN diagonal matrix
    given by the area of x nd y data spacing.
	rho0: numpy array - initial equivalent
    layer property estimative.
	itmax: scalar - number of iterations

    output
    m_new: numpy array - final equivalent
    layer property estimative.
    '''
    m_new = np.copy(rho0)
    for i in range (itmax):
        res = (data - A.dot(m_new))
        delta_m = res/diagonal_A
        m_new += delta_m
    return m_new
    
def fast_loop_bccb(cev,shape,N,data,diagonal_A,rho0,itmax):
    '''
    Solves the linear inversion through a iterative method.

    input
	BTTB: numpy array - first line os sensibility matrix.
	shape: tuple - grid size.
	N: scalar - number of observation points.
    data: numpy array - the gravity disturbance.
    diagonal_A: escalar - NxN diagonal matrix
    given by the area of x nd y data spacing.

    output
    m_new: numpy array - final equivalent
    layer property estimative.
    '''
    m_new = np.copy(rho0)
    for i in range (itmax):
        gzp = fast_forward_bccb(shape,N,m_new,cev)
        res = (data - gzp)
        delta_m = res/diagonal_A
        m_new += delta_m
    return m_new
    
def fast_forward_bccb(shape,N,p,cev):
    '''
	Calculate the forward problem at each iteration
    taking advantage of the BTTB (Block-Toeplitz
    Toreplitz-Block) structures.

    input
    shape: tuple - grid size.
    N: scalar - number of observation points.
    p: numpy array - equivalent layer property
	estimative.
	BTTB: numpy array - first line os sensibility matrix.

    output
    gzp: numpy array - the predicted data.
    '''
    v = np.zeros(4*N, dtype='complex128')
    for i in range (shape[0]):
        v[shape[1]*(2*i):shape[1]*(2*i+2)] = np.concatenate((p[shape[1]*(i):shape[1]*(i+1)], np.zeros(shape[1])), axis=None)
    
    v = v.reshape(2*shape[0],2*shape[1]).T
    gzp = np.fft.ifft2(np.fft.fft2(v)*cev)
    gzp = np.ravel(np.real(gzp[:shape[1],:shape[0]]).T)
    return gzp