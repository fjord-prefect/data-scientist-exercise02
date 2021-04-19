import os, re
import json

import numpy as np
import pandas as pd

import datetime
import xml.etree.ElementTree as ET

def variable_descriptions(df):
    desc_str = pd.concat([df.describe().T,(df=='').sum(axis=0).rename('blank_str')], axis=1)
    df = df.apply(pd.to_numeric, errors='ignore')
    desc_float = df.describe().T
    desc = desc_str.merge(desc_float, how='left', left_index=True, right_index=True)
    return desc

def dataset():
    jsons = [i for i in os.listdir('data') if 'Narrative' in i]

    path_to_xml_file = 'data/AviationData.xml'

    tree = ET.parse(path_to_xml_file)

    data = []
    for el in tree.iterfind('./*'):
        for i in el.iterfind('*'):
            data.append(dict(i.items()))

    df_base = pd.DataFrame(data)

    segments = []
    for i in jsons:
        segments.append(pd.DataFrame.from_dict(json.load(open('data/'+i))['data']))
    df_nar = pd.DataFrame(data = np.vstack(segments), columns = segments[0].columns)
    
    df = df_nar.merge(df_base, how='inner', on='EventId')
    df.index = pd.to_datetime(df['EventDate'], infer_datetime_format=True)
    df = df.sort_index().loc[datetime.date(year=1982,month=1,day=1):]
    
    return variable_descriptions(df), df.apply(pd.to_numeric, errors='ignore')