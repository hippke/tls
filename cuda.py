@cuda.jit
def itr_cuda(data, a, s, result):
    i = cuda.grid(1)
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = tx + bx * bw
    xstride = cuda.gridsize(1)
    for j in range(i, s.shape[0], xstride):
        while k < len(data)-s.shape[1] and l < s.shape[1]+1:
            result[l,k] += (data[k+l] - s[j,l])**2 * a[k+l]
