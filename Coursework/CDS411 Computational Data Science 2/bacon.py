# bacon.py
# JM Kinser
# (C) Jason Kinser 2022.
# This code is intended for non-commercial use in courses at George Mason Univ.
# Their are no guarantees with this code
# Commercial and/or non-academic use prohibited without written permission from the author.


#%%
import numpy as np
import floyd
import movies3 as mvs
#%%
## Create G
def MakeG( isin ):
    ## list of unique aids + sort
    mat = np.array( isin )
    t = mat[:,2]+0
    aids = np.array(list(set(t)))
    ## allocate G
    N = len(aids)
    G = np.zeros((N+1,N+1))
    ## for each actor
    for i in aids:
        ## get mids
        mids = mvs.MidsFromAid(isin, i)
        ## aids from actor's movies
        aids2 = mvs.AidsFromMids( isin, mids )
        # populate G
        if i==771:
            print(aids2)
        for a in aids2:
            G[i-1, a-1] = 1
    ## return G
    print(G[770,770])
    return G
#%%
def TestFloyd():
    G = np.zeros((6,6)) + 999999999
    G[0,1] = G[1,0] = 4
    G[0,2] = G[2,0] = 2
    G[1,2] = G[2,1] = 5
    G[1,3] = G[3,1] = 10
    G[2,4] = G[4,2] = 3
    G[3,4] = G[4,3] = 4
    G[3,5] = G[5,3] = 11
    for i in range(6):
        G[i,i] = 0
    p = np.zeros((6,6))
    for i in range( 6 ):
        p[i] = i
    f,p = floyd.FastFloydP(G,p)
    return f,p
#%%
# make an image out of G
def RunFloyd( G ):
    # need large values for disconnects
    GG = np.zeros( G.shape )
    GG = G + (1-G)*9999999
    ndx = np.indices( GG.shape )
    pp = (G * ndx[0]).astype(int)
    g,p = floyd.FastFloydP( GG,pp )
    return g,p


