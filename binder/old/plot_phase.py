
import pandas as pd
import click
import os,glob

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def plot_prbsscan(df):
    prbs_data = df.rename(columns={'CH_0':0,
                                   'CH_1':1,
                                   'CH_2':2,
                                   'CH_3':3,
                                   'CH_4':4,
                                   'CH_5':5,
                                   'CH_6':6,
                                   'CH_7':7,
                                   'CH_8':8,
                                   'CH_9':9,
                                   'CH_10':10,
                                   'CH_11':11})
    prbs_data = np.array(prbs_data)
    a,b = np.meshgrid(np.arange(12), np.arange(15))
    plt.hist2d(a.flatten(), b.flatten(), weights=prbs_data.flatten(), bins=(np.arange(13), np.arange(16)), cmap='bwr')
    plt.colorbar()
    plt.ylabel('Phase Select Setting')
    plt.xlabel('Channel Number')  

def get_phase_scan(timestamp, capselect, voltage="1_20", board="10"):
    fname = f'board_{board}/voltage_{voltage}/{timestamp}/eRx_PhaseScan_CapSelect_{capselect}.csv'
    prbs_data = pd.read_csv(fname, header=None)
    return prbs_data

def get_phase_from_time(fname):
    try:
        x=np.loadtxt(fname,delimiter=',',dtype=int) 
    except: 
        x=np.ones(15*12,dtype=int).reshape(15,12)*999
    return x

def get_max_width(err_counts):
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
    return np.array(max_width_by_ch),np.array(second_max_width_by_ch)

def get_best_setting(err_counts):
    best_setting_by_ch = []
    counts_window = []
    for i in range(15):
        counts_window.append( err_counts[i] + err_counts[(i-1)%15] + err_counts[(i+1)%15])
    counts_window = np.array(counts_window)
    counts_window[ err_counts>0 ] += 255*3
    y = (err_counts[2:-2]+err_counts[1:-3]+err_counts[3:-1]+err_counts[4:] + err_counts[:-4])
    y[ err_counts[2:-2]>0 ] += 2555
    best_setting = y.argmin(axis=0)+2
    for ch in range(12):
        best_setting_by_ch.append(best_setting[ch])
    return np.array(best_setting_by_ch)

@click.command()
@click.option('--board', default="10", help='ASIC')
@click.option('--voltage', default="1_20", help='voltage')
def scan_timestamp(board, voltage):
    timestamps = os.listdir(f'board_{board}/voltage_{voltage}/')
    df=pd.DataFrame({'Timestamp':timestamps})
    df['Date']=df['Timestamp'].apply(lambda x: pd.to_datetime(x,format="%d%b_%H%M%S").replace(year=2022))
    df=df.loc[df.Date>'2022-09-26 12']
    df.set_index('Date',inplace=True)
    df.sort_index(inplace=True)

    # allowed_capSel = np.array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
    #                              13,  14,  15,  24,  25,  26,  27,  28,  29,  30,  31,  56,  57,
    #                              58,  59,  60,  61,  62,  63, 120, 121, 122, 123, 124, 125, 126,
    #                              127, 248, 249, 250, 251, 252, 253, 254, 255, 504, 505, 506, 507,
    #                              508, 509, 510, 511])
    allowed_capSel = np.array([25,26,28,29,30,31])

    for i,capSel in enumerate(allowed_capSel):
        df_cap = df.copy()
        df_cap['CapSel'] = capSel
        df_cap['phase_scan'] = df_cap['Timestamp'].apply(lambda x: get_phase_from_time(f'board_{board}/voltage_{voltage}/{x}/eRx_PhaseScan_CapSelect_{capSel}.csv'))
        err_counts = df_cap['phase_scan'].values
        df_cap['max_width'] = df_cap['phase_scan'].apply(lambda x: get_max_width(x)[0])
        # df_cap['2nd_max_width'] = df_cap['phase_scan'].apply(lambda x: get_max_width(x.T)[1])
        df_cap['best_phase'] = df_cap['phase_scan'].apply(lambda x: get_best_setting(x))
        if i==0:
            df_new = df_cap.copy()
        else:
            df_new = pd.concat([df_cap, df_new])

    # initialize dictionaries
    capSel_by_time = {}
    for timestamp in timestamps:
        capSels = [int(x.split('_')[-1][:-4]) for x in glob.glob(f'board_{board}/voltage_{voltage}/{timestamp}/eRx_PhaseScan_CapSelect_*.csv')]
        capSel_by_time[timestamp] = [int(x) for x in capSels]

    ch = 0
    capSel = 28
    var = 'max_width'
    arr = df_new.loc[(df_new['CapSel'] == capSel)]
    timestamps = arr.index.to_numpy()
    values = arr[var].values
    val = np.array([list(arr) for arr in values]).T

    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.CMS)

    enu_timestamps = np.arange(np.size(timestamps))
    
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    a,b = np.meshgrid(enu_timestamps, np.arange(12))
    print(val.flatten())
    print(a.flatten())
    print(b.flatten())
    _,_,_,im = ax.hist2d(a.flatten(), b.flatten(), weights=val.flatten(), bins=(enu_timestamps, np.arange(12)))
    cbar = fig.colorbar(im, ax=ax, orientation='vertical')
    # ax.plot(timestamps, val[ch])
    # ax.legend(loc='upper left',  bbox_to_anchor=(1.05, 1))
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('eRx')
    ax.set_title(f'CapSel {capSel}')
    fig.tight_layout()
    fig.savefig('test.png')

scan_timestamp()
