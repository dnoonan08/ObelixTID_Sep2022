import os
import pandas as pd
import numpy as np
from utils import get_phase,get_max_width,get_best_setting,cap_bank_vals,get_phase_dict
from utils import get_tid,get_allowed_index_map,start_times,date_times
import click
import re

def parse_scans(asic,voltages=['1_20','1_08','1_32']):
    datecut = date_times[asic]
    directory = f"../PhaseScans/board_{asic}"
    scanDirs = os.listdir(f"{directory}/voltage_1_20/")
    
    df = pd.DataFrame({'timestamp':scanDirs})
    df['time'] = df['timestamp'].apply(lambda x: pd.to_datetime(x,format="%d%b_%H%M%S").replace(year=2022))
    
    df = df.loc[df.time>datecut]
    df.set_index('time',inplace=True)
    df.sort_index(inplace=True)
    df['TID'] = get_tid(df.index.values,asic)
    # df=df.loc[df.TID>0]

    scan_strs = {
        'eRx': 'PhaseScan',
        'eTx': 'DelayScan',
    }
    # careful that we are only choosing a certain number of values
    for i,val in enumerate(cap_bank_vals):
        df_cap = df.copy()
        df_cap['capBank_val'] = val
        for voltage in voltages:
            df_cap[f'phases_trackMode0_{voltage}'] = pd.Series(df_cap['timestamp'].apply(lambda x: get_phase_dict(f'../PhaseScans/board_{asic}/voltage_{voltage}/{x}/phaseSelect_TrackMode0.txt',val=val)))
            df_cap[f'phases_trackMode1_{voltage}'] = pd.Series(df_cap['timestamp'].apply(lambda x: get_phase_dict(f'../PhaseScans/board_{asic}/voltage_{voltage}/{x}/phaseSelect_TrackMode1.txt',val=val)))
            for scan in ['eRx','eTx']:
                df_cap[f'{scan}Scan_{voltage}'] = pd.Series(df_cap['timestamp'].apply(lambda x: get_phase(scan, f'{directory}/voltage_{voltage}/{x}/{scan}_{scan_strs[scan]}_CapSelect_{val}.csv')))
                df_cap[f'{scan}Best_{voltage}'] = df_cap[f'{scan}Scan_{voltage}'].apply(lambda x: get_best_setting(x))
                df_cap[f'{scan}MaxWidth_{voltage}'] = df_cap[f'{scan}Scan_{voltage}'].apply(lambda x: get_max_width(x)[0])
                df_cap[f'{scan}2ndMaxWidth_{voltage}'] = df_cap[f'{scan}Scan_{voltage}'].apply(lambda x: get_max_width(x)[1])

        if i==0:
            df_new = df_cap.copy()
        else:
            df_new = pd.concat([df_cap, df_new])
            
    df_new = df_new.reset_index()
    return df_new

def parse_log(fname,asic):
    _f = open(f'../logs/{fname}')
    _fLines = _f.read()
    
    def find_match(re_string):
        return np.array(re.findall(re_string,_fLines))

    def extract_pll_info(pll_set_matches):
        """
        Extract pll scan info
        """
        pll_dates = []
        pll_settings = []
        pll_voltages = []
        pll_pusm = []
        pll_good = []
        pll_third = []
    
        pusm = []
        settings = []
        good = []
        
        for m in pll_set_matches:
            if not m[0]=='':
                # the rows that start with the date only contain the time at the start of the scan and voltage
                pll_dates.append(m[0]+'-'+m[1])
                pll_voltages.append(float(m[2]))
                if len(pusm) > 0:
                    pll_pusm.append(np.array(pusm))
                    pll_settings.append(np.array(settings))
                    pll_good.append(np.array(good))
                    pll_third.append(good[int(len(good)/3)])
                    pusm = []
                    settings = []
                    good = []
            else:
                pusm.append(int(m[-1]))
                settings.append(int(m[-3]))
                if int(m[-1]) == 9:
                    good.append(int(m[-3]))
                
        # use this for the last row
        pll_pusm.append(np.array(pusm))
        pll_settings.append(np.array(settings))
        pll_good.append(np.array(good))
        pll_third.append(good[int(len(good)/3)])
                
        # apply mask
        return pll_dates,pll_settings,np.array(pll_voltages),pll_pusm,pll_good,pll_third

    def voltage_stamp(voltages,arr):
        """
        Fixes timestamp for all voltages (mostly needed for chip 10)
        - Assumes that at 1.2 we end the scan
        - Builds an array of timestamps for all voltages
        - And returns an array of booleans for all voltages,
          where True means that it has a timestamp
        """
        diff = np.split(voltages, np.where(voltages==1.2)[0]+1)
        new_arr = []
        for i,d in enumerate(diff):
            try:
                for j in d:
                    new_arr.append(arr[i])
            except:
                new_arr.append('0')
        return new_arr

    def setting_stamp(pll_voltage,pll_voltages,arr):
        pll_voltage_copy = pll_voltage.astype(np.float64)
        new_arr = []
        for voltage in pll_voltages.astype(np.float64):
            if voltage==pll_voltage_copy[0]:
                try:
                    new_arr.append(int(arr[0]))
                    arr = np.delete(arr,0)
                    pll_voltage_copy = np.delete(pll_voltage_copy,0)
                except:
                    new_arr.append(0)
            else:
                new_arr.append(0)
        return new_arr
    
    # readings on voltage,current,temp and resistance and error counts
    reading_matches = find_match(r"([019]*)-(.*) INFO   Power: On, Voltage: (.*) V, Current: (.*) A, Temp: (.*) C, Res.: (.*) Ohms\n\1.* Word count (\d*), error count (\d*)")
    reading_dates = pd.to_datetime([x[0]+'-'+x[1] for x in reading_matches[:,:2]])
    reading_voltage = np.float32(reading_matches[:,2])
    reading_current = np.float32(reading_matches[:,3])
    reading_temp = np.float32(reading_matches[:,4])
    reading_resistance = np.float32(reading_matches[:,5]) 
    reading_words = np.float32(reading_matches[:,6])
    reading_errors = np.float32(reading_matches[:,7])
    reading_error_rate = reading_errors/reading_words
    reading_tid = get_tid(reading_dates.values,asic)
    
    # replace bad-readings
    bad_readings = (reading_voltage==-1)&(reading_current==-1)
    reading_voltage[bad_readings]=np.nan
    reading_current[bad_readings]=np.nan
    bad_readings = (reading_temp==-1)&(reading_resistance==-1)
    reading_temp[bad_readings]=np.nan
    reading_resistance[bad_readings]=np.nan

    # check if previous readings match an i2c transaction
    i2c_matches = find_match(r"([019]*)-.* RW M.*\n\1-(.*) INFO   Power: On, Voltage: (.*) V, Current: (.*) A, Temp: (.*) C, Res.: (.*) Ohms\n\1.* Word count (\d*), error count (\d*)")
    i2c_matches_gpiberror = find_match(r"([019]*)-.* RW M.*\n\1.*ERROR  Unable to reconnect to GPIB.*\n\1-(.*) INFO   Power: On, Voltage: (.*) V, Current: (.*) A, Temp: (.*) C, Res.: (.*) Ohms\n\1.* Word count (\d*), error count (\d*)")
    if len(i2c_matches)>0:
        i2c_dates = pd.to_datetime([x[0]+'-'+x[1] for x in i2c_matches[:,:2]])
        is_i2c = reading_dates.isin(i2c_dates)
    else:
        is_i2c = np.full(reading_dates.shape, False)
    if len(i2c_matches_gpiberror)>0:
        i2c_dates_gpib = pd.to_datetime([x[0]+'-'+x[1] for x in i2c_matches_gpiberror[:,:2]])
        is_i2c = np.bitwise_or(reading_dates.isin(i2c_dates) , reading_dates.isin(i2c_dates_gpib))
    
    # ro mismatches
    ro_matches = find_match(r"([019]*)-(.*) ERROR  RO Mismatches: {'ASIC': (.*)}")
    ro_dates = pd.to_datetime([x[0]+'-'+x[1] for x in ro_matches[:,:2]])

    # look at header counter mismatch errors
    y=np.zeros(len(ro_dates)*12).reshape(-1,12)
    for i in range(12):
        for j in range(len(ro_dates)):
            try:
                y[j][i]=eval(ro_matches[j,2])[f'CH_ALIGNER_{i}INPUT_ALL']['hdr_mm_cntr'][1]
            except:
                y[j][i]=np.nan
    df_i2c=pd.DataFrame(y,columns=[f'CH{i}_hdr_mm' for i in range(12)],index=ro_dates)
    df_i2c=df_i2c.fillna(method='ffill').fillna(0)

    # good pll settings
    pll_set_matches = find_match("([019]+)-(.*) INFO   Good PLL settings V=(.*):|([019]+)-(.*) INFO.     CapSel=(\d+), V=(.*), PUSM=(\d)")
    pll_info = extract_pll_info(pll_set_matches)
    pll_dates,pll_settings,pll_voltages,pll_pusm,pll_good,pll_third = pll_info
    pll_dates = pd.to_datetime(pll_dates)

    # pll setting that was actually used
    pll_good_matches = find_match(r'([019]+)-(.*) INFO   Setting PLL VCO CapSelect to (\d*) at V=(.*) with phaseSelect settings of (\d*),(\d*),(\d*),(\d*),(\d*),(\d*),(\d*),(\d*),(\d*),(\d*),(\d*),(\d*)')
    if len(pll_good_matches)==0:
        pll_good_matches = find_match(r'09-(.*) INFO   Setting PLL VCO CapSelect to (\d*) with phaseSelect settings of (\d*),(\d*),(\d*),(\d*),(\d*),(\d*),(\d*),(\d*),(\d*),(\d*),(\d*),(\d*)')
        pll_setting = pll_good_matches[:,1]
        pll_setting_phase = pll_good_matches[:,2:].tolist()
        pll_voltage = pll_voltages
    else:
        pll_setting = pll_good_matches[:,2]
        pll_setting_phase = pll_good_matches[:,4:].tolist()
        pll_voltage = pll_good_matches[:,3]

    # pll scans timestamp
    pll_timestamp_matches = find_match("([019]+)-(.*) INFO   Starting Power Scans \( timestamp (.*) \)")
    timestamp = pll_timestamp_matches[:,2]

    # deal with weird cases
    if len(timestamp)!=len(pll_info[2]):
        timestamp = voltage_stamp(pll_info[2],timestamp)
        used = np.int32(voltage_stamp(pll_info[2],pll_setting))

    if len(pll_setting) == len(pll_voltages):
        used = np.int32(pll_setting)
    else:
        if asic=='9':
            used = setting_stamp(pll_voltage,pll_voltages,pll_setting)

    # PLL dataframe
    df_pll = pd.DataFrame(pll_voltages, columns=['voltage'])
    df_pll['timestamp'] = timestamp
    df_pll['cb_settings'] = pd.Series(pll_good)
    df_pll['cb_pusm'] = pd.Series(pll_pusm)
    df_pll['ngood'] = df_pll.cb_settings.apply(lambda x: len(x))
    df_pll['maxgood'] = df_pll.cb_settings.apply(lambda x: max(x))
    df_pll['mingood'] = df_pll.cb_settings.apply(lambda x: min(x))
    df_pll['maxgood_index'] = df_pll.cb_settings.apply(lambda x: get_allowed_index_map(max(x)))
    df_pll['mingood_index'] = df_pll.cb_settings.apply(lambda x: get_allowed_index_map(min(x)))
    df_pll['third'] = pll_third
    df_pll['third_index'] = df_pll.third.apply(lambda x: get_allowed_index_map(x))

    df_pll['used'] = used
    df_pll['used_index'] = df_pll.used.apply(lambda x: get_allowed_index_map(x))
    if len(pll_setting_phase) == len(pll_third):
        df_pll['used_phases'] = pd.Series([np.int32(l) for l in pll_setting_phase])
    else:
        # triplicate because we did not save the phases for the other voltages
        df_pll['used_phases'] = pd.Series(list(np.repeat(pll_setting_phase,3,axis=0)))

    df_pll['time'] = pll_dates
    df_pll['TID'] = get_tid(pll_dates.values,asic)

    # fill dataframe
    output = {}
    output['time']=reading_dates
    output['TID']=reading_tid
    output['voltage']=reading_voltage
    output['current']=reading_current
    output['temp']=reading_temp
    output['rtd']=reading_resistance
    output['nComp']=reading_words
    output['nErr']=reading_errors
    output['errRate']=reading_error_rate
    output['isI2C']=is_i2c
    df = pd.DataFrame.from_dict(output)

    return  df,df_pll,df_i2c

@click.command()
@click.option('--asic', required=True, help='ASIC, choices 9 or 10')
def parse(asic):
    log_strs = {
        '10': [
            #'logFile_Chip10_CoolDown.log',
            #'logFile_Chip10_StartIrradiation.log',
            #'logFile_Chip10_StartIrradiation_Sept28_05h28.log',
            #'logFile_Chip10_StartIrradiation_Sept28_05h58.log',
            #'logFile_Chip10_StartIrradiation_Sept28_07h05.log',
            #'logFile_Chip10_StartIrradiation_Sept28_07h23.log',
            #'logFile_Chip10_StartIrradiation_Sept28_14h14.log',
            #'logFile_Chip10_StartIrradiation_Sept28_16h43.log',
            #'logFile_Chip10_StartIrradiation_Sept28_16h59.log',
            #'logFile_Chip10_StartIrradiation_Sept28_17h19.log',
            'logFile_Chip10_StartIrradiation_Sept28_20h16.log',
        ],
        '9': [
            'logFile_Chip09_CoolDown.log',
            'logFile_Chip09_StartIrradiation.log',
            'logFile_Chip09_17h48_StartIrradiation.log',
            'logFile_Chip09_Oct2_09h48_StartIrradiation.log',
            'logFile_Chip09_Oct2_12h26_StartIrradiation.log',
            'logFile_Chip09_Oct2_12h58_StartIrradiation.log',
            'logFile_Chip09_Oct2_19h03_StartIrradiation.log',
            'logFile_Chip09_Oct3_11h33_StartIrradiation.log',
            'logFile_Chip09_Oct3_12h16_StartIrradiation.log',
        ]
    }

    parsed_logs = [parse_log(lname,asic) for lname in log_strs[asic]]
    dfs = {}
    dfs['irr'] = pd.concat([p[0] for p in parsed_logs], ignore_index=True)
    dfs['pll'] = pd.concat([p[1] for p in parsed_logs], ignore_index=True)
    dfs['i2c'] = pd.concat([p[2] for p in parsed_logs], ignore_index=True)
    click.echo(f'Parsing scans')
    dfs['scans'] = parse_scans(asic)

    for key,df in dfs.items():
        click.echo(f'Saving {key} dataframe')
        # need to preserve dtypes - prepend to the top
        df.loc[-1] = df.dtypes
        df.index = df.index + 1
        df.sort_index(inplace=True)
        # then save
        df.to_csv(f'../data/{key}_{asic}.csv', index=False) 
    
if __name__ == '__main__':
    parse()
