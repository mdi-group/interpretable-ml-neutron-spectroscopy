import numpy as np
#import pandas as ps

def readMat(filename='../../data/Rb2MnF4/rb2mnf4_resolution_function.txt'):
    fid = open(filename,'r')
#    header = fid.readline()
    
    result = {}
    
    for line in fid.readlines():
        values = [float(s) for s in line.split()]
        assert(len(values)==12),'Input length error'
        q = [round(v,1) for v in values[0:2]]
        e = round(values[2],1)
        covar = values[3:]
        assert(abs(covar[1]-covar[3])<1e-10),'This is not symmetric: {} != {}'.format(covar[1],covar[3])
        assert(abs(covar[2]-covar[6])<1e-10),'This is not symmetric: {} != {}'.format(covar[2],covar[6])
        assert(abs(covar[5]-covar[7])<1e-10),'This is not symmetric: {} != {}'.format(covar[5],covar[7])
        if(e not in result):
            result[e] = {}
        if(q[0] not in result[e]):
            result[e][q[0]] = {}
        assert(q[1] not in result[e][q[0]]),'Repeated entry Error'
        result[e][q[0]][q[1]] = [covar[i] for i in [0,1,2,4,5,8]] # Only store upper triangle
    return result
    

def covariance(table,coord=[0,0,0]):
    # find 8 points around the point to get covariance for
    gridCoord = [[None,None] for i in range(len(coord))]
    tmp = table
    for i,v in enumerate(coord):
        for k in sorted(tmp.keys()):
            if(gridCoord[i][1] == None):
                if(k > v):
                    gridCoord[i][1] = k
                else:
                    gridCoord[i][0] = k
                
        if(gridCoord[i][0] == None):
            gridCoord[i][0] = gridCoord[i][1]
            gridCoord[i][1] = None
        
        if(i+1<len(coord)):
            tmp = tmp[gridCoord[i][0]]
    # create a weighted average of these
    gridDist = [[c-g[0],g[1]-c if g[1]!=None else None] for g,c in zip(gridCoord,coord)]
    to_add = []
    total_dist = 0
    for i,e in enumerate(gridCoord[0]):
        if(e==None):
            continue;
        to_add1 = []
        total_dist1 = 0
            
        for j,qh in enumerate(gridCoord[1]):
            if(qh==None):
                continue;
            to_add2 = []
            total_dist2 = 0
            for k,qk in enumerate(gridCoord[2]):
                if(qk==None):
                    continue;
                to_add2 += [[np.array(table[e][qh][qk]),gridDist[2][k]]]
                total_dist2 += gridDist[2][k]
            if total_dist2==0:
                total_dist2=1
            tmp = np.zeros(to_add2[0][0].shape)
            for k in to_add2:
                tmp += k[0]*(total_dist2-k[1])/total_dist2
            to_add1 += [[tmp,gridDist[1][j]]]
            total_dist1 += gridDist[1][j]
         
        if total_dist1==0:
            total_dist1=1
        tmp = np.zeros(to_add1[0][0].shape)
        for j in to_add1:
            tmp += j[0]*(total_dist1-j[1])/total_dist1
        to_add += [[tmp,gridDist[0][i]]]
        total_dist += gridDist[0][i]
    
    if total_dist==0:
        total_dist=1
    covarList = np.zeros(to_add[0][0].shape)
    for i in to_add:
        covarList += i[0]*(total_dist-i[1])/total_dist
        
    mat = np.zeros((len(coord),len(coord)))
    #todo - rewrite in a general method 
    mat[0,0] = covarList[5] #exe
    mat[0,1] = covarList[2] #exqh
    mat[1,0] = covarList[2] #qhxe
    mat[0,2] = covarList[4] #exqk
    mat[2,0] = covarList[4] #qkxe
    mat[1,1] = covarList[0] #qhxqh
    mat[1,2] = covarList[1] #qhxqk
    mat[2,1] = covarList[1] #qkxqh
    mat[2,2] = covarList[3] #qkxqk
    
    #print(mat)
    return mat
