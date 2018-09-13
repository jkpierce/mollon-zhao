# random shape generation from fourier coefficients using the mollon and zhao algorithm 
import numpy as np 

# i was worse at python when I wrote this stuff 

def alphamat(verts): 
    """generate the angle matrix from the set of geodesic verticies"""
    n = len(verts)
    alpha = np.zeros((n,n))
    for nv,v in enumerate(verts):
        for nw,w in enumerate(verts):
            dott = np.dot(v,w)    
            dott = np.clip(dott,-1,1) #make sure it doesn't explode the arccos due to machine precision
            alpha[nv,nw] = np.arccos(dott)
    return alpha

def C_mat(K,alpha):
    if K>=2:
        C = lambda a,K: 2*np.cos(K*a)
        Cv = np.vectorize(C)
        return Cv(alpha,K)
    
    
def rho_prime(cmat,K): #eq 15 I hope 
    from scipy import linalg as LA
    evals, evecs = LA.eig(cmat)
    
    #now sort both arrays by the values of evals highest to lowest and take the first 2K+1 elements
    order = evals.argsort()[::-1] #sort in descending order 
    n = len(evals)
    evals = evals[order][:2*K+1].real
    evecs = np.array([evecs[:,i] for i in range(n)])[order][:2*K+1] #so here is an array with the eigenvecs in order
    P = np.column_stack(evecs) #here is the matrix composed of dominant eigenvectors 
    L = np.diag(evals) #and the matrix composed of dominant eigenvalues
    out = np.dot(P,np.dot(L,P.T)).real #and here is eq. 13 
    ep = np.amax(out) 
    return out/ep #this is [rho']_K


def random_realization(Cp):
    #find the eigensystem of the correlation matrix -- hopefully positive deifnite
    import scipy.linalg as LA
    evals, evecs = LA.eig(Cp)
    n=len(evals) 
    #define the mean 
    mu = np.ones(n) #mean is 1 
    #generate array of standard normal random variables 
    xi = np.array([np.random.normal(0,1) for i in range(n)])
    #calculate the array of radii 
    r =  mu + np.sum([xi[i]*np.sqrt(evals[i])*evecs[:,i] for i in range(n)],axis=0).real
    rmax = np.amax(r)
    r=r/rmax
    return r

def total_autocorr(rhoprimes,weights):
    """this generates the autocorrelation function Cp"""
    return np.sum([rhoprimes[i]*w for i,w in enumerate(weights)],axis=0)

def D(k): 
    """example fourier coefficients from the figure in the mollon zhao paper"""
    if k==0: 
        return 1
    elif k==1:
        return 0
    elif k==2:
        return 0.075
    elif 3<=k<=64: 
        return 2**(-1.6*np.log2(k/2)+np.log2(0.075))
    elif k>64:
        kp = 128-k
        return 2**(-1.6*np.log2(kp/2)+np.log2(0.075))
    
def gen_weights(D,k_range = np.arange(2,65)):
    """generate the weights for the expansion of the shape from fourier coefficients"""
    return [2*D(k)**2 for k in k_range]  

def generate_stones(N,Cp,verts): 
    stones = []
    for i in range(N): 
        rando = random_realization(Cp)
        stone = np.array([v*rando[i] for i,v in enumerate(verts)])
        print('particle of random shape # ',i, ' created')
        stones.append(stone)
    return stones