def itr_cuda(data, dys, signals, chi2map):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bz = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y
    bd = cuda.blockDim.z
    signal_trial = tx + bx * bw
    phase_position = ty + by * bh
    in_transit_point = tz + bz * hd
    if signal_trial < signals.shape[0] and \
    phase_position < len(data)-signals.shape[1]+1 and \
    in_transit_point < signals.shape[1]:
        datapoint = data[phase_position+in_transit_point]
        signal = signals[signal_trial, in_transit_point]
        error = dys[phase_position+in_transit_point]
        residual = ((datapoint - signal)**2 * error)
        chi2map[signal_trial, phase_position] += residual
