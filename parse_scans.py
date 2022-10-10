import os
import pandas as pd
import numpy as np
from utils import get_phase,get_max_width,get_best_setting,cap_bank_vals
from utils import get_tid

def parse_scans(board,datecut='2022-09-26 14'):
    directory = f"PhaseScans/board_{board}"
    scanDirs = os.listdir(f"{directory}/voltage_1_20")
    
    df = pd.DataFrame({'timestamp':scanDirs})
    df['time'] = df['timestamp'].apply(lambda x: pd.to_datetime(x,format="%d%b_%H%M%S").replace(year=2022))
    
    df = df.loc[df.time>datecut]
    df.set_index('time',inplace=True)
    df.sort_index(inplace=True)
    
    df['TID'] = get_tid(df.index.values,board)
    # df=df.loc[df.TID>0]

    scan_strs = {
        'eRx': 'PhaseScan',
        'eTx': 'DelayScan',
    }
    for i,val in enumerate(cap_bank_vals):
        df_cap = df.copy()
        df_cap['capBank_val'] = val
        for voltage in ['1_20','1_08','1_32']:
            for scan in ['eRx','eTx']:
                df_cap[f'{scan}Scan_{voltage}'] = df_cap['timestamp'].apply(lambda x: get_phase(scan, f'{directory}/voltage_{voltage}/{x}/{scan}_{scan_strs[scan]}_CapSelect_{val}.csv'))
                df_cap[f'{scan}Best_{voltage}'] = df_cap[f'{scan}Scan_{voltage}'].apply(lambda x: get_best_setting(x))
                df_cap[f'{scan}MaxWidth_{voltage}'] = df_cap[f'{scan}Scan_{voltage}'].apply(lambda x: get_max_width(x)[0])
                df_cap[f'{scan}2ndMaxWidth_{voltage}'] = df_cap[f'{scan}Scan_{voltage}'].apply(lambda x: get_max_width(x)[1])
                
        if i==0:
            df_new = df_cap.copy()
        else:
            df_new = pd.concat([df_cap, df_new])
            
    # df_new = df_new.reset_index()
    return df_new
