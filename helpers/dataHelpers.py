import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def getData(all = False):
    if all:
        dataBlocks = []
        for num in range(1,112):
            df = pd.read_csv("dane/halfhourly_dataset/block_"+str(num)+".csv")
            dataBlocks.append(df)
        data = pd.concat(dataBlocks, ignore_index=True)
        return data
    else:
        df = pd.read_csv("dane/halfhourly_dataset/block_0.csv")
        data = df.loc[df["LCLid"] == "MAC000002"]
        return data

def prepareData():
    data = getData()

    data['date_time'] = pd.to_datetime(data['tstp'])
    data = data.dropna(subset=['energy(kWh/hh)'])
    data = data[['LCLid', 'date_time', 'energy(kWh/hh)']]
    data['year'] = data['date_time'].apply(lambda x: x.year)
    data['quarter'] = data['date_time'].apply(lambda x: x.quarter)
    data['month'] = data['date_time'].apply(lambda x: x.month)
    data['day'] = data['date_time'].apply(lambda x: x.day)
    data['hour'] = data['date_time'].apply(lambda x: x.hour)
    data['minute'] = data['date_time'].apply(lambda x: x.minute)
    data = data.loc[:,['date_time','energy(kWh/hh)', 'year', 'quarter', 'month', 'day', 'hour', 'minute']]
    data.sort_values('date_time', inplace=True, ascending=True)
    data = data.reset_index(drop=True)
    data['weekday'] = data['date_time'].apply(lambda x: x.weekday() < 5).astype(int)

    return data