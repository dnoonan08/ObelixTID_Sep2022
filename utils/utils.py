import numpy as np
import os

allowed_cap_bank_vals=np.array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
                                  13,  14,  15,  24,  25,  26,  27,  28,  29,  30,  31,  56,  57,
                                  58,  59,  60,  61,  62,  63, 120, 121, 122, 123, 124, 125, 126,
                                  127, 248, 249, 250, 251, 252, 253, 254, 255, 504, 505, 506, 507,
                                  508, 509, 510, 511])
def get_allowed_index_map(cap_bank):
    allowed_map=np.zeros(512,dtype=int)
    for i,j in enumerate(allowed_cap_bank_vals):
        allowed_map[j]=i
    return allowed_map[cap_bank]

cap_bank_vals  = np.array([  # 0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,
                             11,  12, 13,  14,  15,  24,  25,  26,  27,  28,  29,  30,  31,  56,  57,
                             58,  59,  60,  61,  62,  63, 120, 121, 122, 123, 124, 125, 126,
                             127])
vdict = {
    '1_20': 1.20,
    '1_08': 1.08,
    '1_32': 1.32,
}
start_times = {
    '10': np.datetime64('2022-09-26 14:45'),
    '9': np.datetime64('2022-09-30 08:20'),
}
rates = {
    '10': 5.51,
    '9': 8.85,
}
date_times = {
    '10': '2022-09-26 12',
    '9': '2022-09-29 12',
}

def get_phase_dict(fname,val):
    import pickle
    cb_settings = []
    used_phase = []
    if os.path.exists(fname):
        with open(fname) as f:
            data = f.readlines()
        for l in data:
            cb = int(l.split(':')[0].replace('{',''))
            cb_settings.append(cb)
            if cb == val:
                x = l.split(':')[1].rstrip().strip().replace('array(','').replace('[','').replace(']),','')
                used_phase = np.fromstring(x, dtype=int,sep=' , ')
    cb_settings = np.array(cb_settings)
    used_phase = np.array(used_phase)
    return used_phase

def plot_tid(tid=False):
    if tid: 
        col = 'TID'
        axtitle = 'TID [Mrad]'
    else: 
        col = 'time'
        axtitle = 'Time'
    return col,axtitle

def get_tid(times,board):
    start_time = start_times[board]
    t0=(times-start_time).astype(int)/3.6e12
    t0[t0<0]=0
    rate = rates[board]
    return t0*rate

def get_phase(scan,fName):
    if scan == 'eTx':
        # eTx scan
        try:
            x=np.loadtxt(fName,delimiter=',',dtype=int) 
        except: 
            x=np.ones(63*13,dtype=int).reshape(63,13)*999
    else:
        # eRx scan
        try:
            x=np.loadtxt(fName,delimiter=',',dtype=int)
        except:
            x=np.ones(15*12,dtype=int).reshape(15,12)*300
    return x

def get_max_width(err_counts):
    def consecutive(data, stepsize=1):
        return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    max_width_by_ch = []
    second_max_width_by_ch = []
    err_wrapped=np.concatenate([err_counts,err_counts[:4]])
    for ch in range(12):
        x = err_wrapped[:,ch]
        phases = consecutive(np.argwhere(x<=1).flatten())
        sizes = [np.size(a) for a in phases]
        max_width = max(sizes)
        sizes.remove(max_width)
        try:
            second_max_width = max(sizes)
        except:
            second_max_width = 0
        max_width_by_ch.append(max_width)
        second_max_width_by_ch.append(second_max_width)
    return np.array([max_width_by_ch,second_max_width_by_ch])

def get_best_setting(err_counts):
    best_setting_by_ch = []
    y = (5*err_counts[2:-2]+
         3*err_counts[1:-3]+
         3*err_counts[3:-1]+
         1*err_counts[4:] + 
         1*err_counts[:-4])
    y[ err_counts[2:-2]>0 ] += 2555
    best_setting = y.argmin(axis=0)+2
    return best_setting
