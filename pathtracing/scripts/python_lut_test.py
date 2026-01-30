from copy import deepcopy

lut = [[0,0,0,0],
       [0,0,0,0],
       [0,0,0,0],
       [0,0,0,0],
       [0.5,0.5,0.5,0.1],
       [0.5,0.5,0.5,0.15],
       [0.5,0.5,0.5,0.16],
       [1,1,1,1]]



def compute_lut_cdf(lut):
    cdf = deepcopy(lut)
    size = len(cdf)

    for i in range(1, size):
        cdf[i][3] += cdf[i - 1][3]
    
    integral = cdf[-1][3]

    for i in range(size):
        cdf[i][3] = (i+1) / size if integral <= 0 else cdf[i][3] / integral

    return cdf


print(lut)
print(compute_lut_cdf(lut))
