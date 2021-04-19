import re,os
import math
import pickle 
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def IncYrPlot(df,title):
    df_ = df.copy()
    df_['yr_str']= [i[-4:] for i in df_['EventDate']]
    ax = df_.groupby('yr_str').count()['EventId'][1:].plot()
    ax.set_xlabel('Year')
    ax.set_title(title)
    plt.show()
    
def scale_latlongs(df, latlong, pop_arr, nonmissing_mask):
    return df[:len(latlong)][nonmissing_mask]['TotalFatalInjuries'].fillna(0)/pop_arr.astype('float')

def latlongs(df):
    latlong = [re.sub('\n','', i) for i in open('tmp/geopoints.txt').readlines()]
    pop = [re.sub('\n','', i) for i in open('tmp/populations.txt').readlines()]

    nonmissing_mask = ((np.array(latlong)!='missing').astype(int) + (np.array(pop)!='nan').astype(int))==2
    
    latlong_arr = np.array(latlong)[nonmissing_mask]
    pop_arr = np.array(pop)[nonmissing_mask]

    longs = np.array([float(i.split()[1]) for i in latlong_arr])
    lats = np.array([float(i.split()[0]) for i in latlong_arr])

    ll_df = pd.DataFrame({'lats':lats, 'longs':longs})
    us_mask = [ll_df['lats'][i]>20 and ll_df['longs'][i]<-40 for i in range(ll_df.shape[0])]
    
    scaled = scale_latlongs(df, latlong, pop_arr, nonmissing_mask)
    
    return lats, longs, us_mask, nonmissing_mask, scaled, pop_arr

def map_plots(longs,lats,scaled,us_mask):
    plt.figure(figsize = (10,8))
    plt.hist2d(longs,lats, bins=150, cmap='hot')
    plt.xlabel('Longitude', fontsize=14)
    plt.ylabel('Latitude', fontsize=14)
    plt.title('Flight Incidents with Fatalities HeatMap', fontsize=17)
    plt.colorbar().set_label('Density of Flight Incidents')
    plt.show()

    plt.figure(figsize = (10,8))
    plt.scatter(longs,lats,c = scaled*10000, s=scaled*10000)
    plt.xlabel("Longitude", fontsize=14)
    plt.ylabel("Latitude", fontsize=14)
    plt.title("Fatalities Down Weighted by Population", fontsize=17)
    plt.colorbar().set_label('Fatalities Down Weighted by Population', fontsize=14)
    plt.show()

    plt.figure(figsize = (10,8))
    plt.hist2d(longs[us_mask],lats[us_mask], bins=150, cmap='hot')
    plt.xlabel('Longitude', fontsize=14)
    plt.ylabel('Latitude', fontsize=14)
    plt.title('Flight Incidents with Fatalities HeatMap', fontsize=17)
    plt.colorbar().set_label('Density of Flight Incidents')
    plt.show()

    plt.figure(figsize = (10,8))
    plt.scatter(longs[us_mask],lats[us_mask],c = scaled[us_mask]*10000, s=scaled[us_mask]*10000)
    plt.xlabel("Longitude", fontsize=14)
    plt.ylabel("Latitude", fontsize=14)
    plt.title("Fatalities Down Weighted by Population", fontsize=17)
    plt.colorbar().set_label('Fatalities Down Weighted by Population', fontsize=14)
    plt.show()

def fatal_rate(df):
    airlines = ['airlines' in i for i in df['AirCarrier'].str.lower()]
    big_companies = df[airlines]
    fatal_inc_rate = big_companies[big_companies['TotalFatalInjuries']>0].shape[0]/big_companies.shape[0]
    print('Total Big Airline Incidents: {} and their Fatal Incident Rate: {}\n'.format(big_companies.shape[0], fatal_inc_rate))

    not_big_companies = df[np.logical_not(airlines)]
    fatal_inc_rate = not_big_companies[not_big_companies['TotalFatalInjuries']>0].shape[0]/not_big_companies.shape[0]
    print('Total NotBig Airline Incidents: {} and their Fatal Incident Rate: {}\n'.format(not_big_companies.shape[0], fatal_inc_rate))

    return None

def fatal_locations(df, nonmissing_mask, pop_arr):
    df_ll = df.iloc[:15000][nonmissing_mask]

    df_ll['Location'] = df_ll['Location'].str.lower()
    df_ll['pop'] = pop_arr.astype(float)

    gb = df_ll[['Location', 'TotalFatalInjuries', 'pop']].groupby(['Location']).sum().reset_index()
    gb['Scaled_Fatalities'] = gb['TotalFatalInjuries']/gb['pop']

    northamer = gb.sort_values(['TotalFatalInjuries'],ascending = False)
    northamer.to_csv('out/NorthAmerican_Flight_Fatalities.csv')
    df['Location'] = df['Location'].str.lower()
    gb = df[['Location', 'TotalFatalInjuries']].groupby(['Location']).sum().reset_index()

    world = gb.sort_values('TotalFatalInjuries',ascending = False)
    world.to_csv('out/World_Flight_Fatalities.csv')

    return northamer, world

def counter(df, keyphrase, fatal_words, nonfatal_words):
    keyphrase = keyphrase.lower()
    df_fatal = df[df['TotalFatalInjuries']>0]
    df_nonfatal = df[df['TotalFatalInjuries']==0]
    
    fatal_sents = [i for i in df_fatal['probable_cause'] if keyphrase in i.lower()]
    nonfatal_sents = [i for i in df_nonfatal['probable_cause'] if keyphrase in i.lower()]
    
    dd = dict({'fatal_sents':fatal_sents, 
                'nonfatal_sents':nonfatal_sents, 
                'total_fatal_sents':len(fatal_sents),
                'total_nonfatal_sents':len(nonfatal_sents)
               })
    
    return len(fatal_sents), len(nonfatal_sents)

def search_top_words(df, keyphrase, fatal_words, nonfatal_words):
    keyphrase = keyphrase.lower()
    fatal_count, nonfatal_count = counter(df, keyphrase, fatal_words, nonfatal_words)

    print('Top Words/Phrases for Fatal Accidents that Contain Your the Phrase: {}'.format(keyphrase))
    s = [i for i in fatal_words.keys() if i not in nonfatal_words and keyphrase in i]
    s.sort()
    for a in s:
        print(a)
    print('\n\n')
    print('Top Words/Phrases for Non-Fatal Accidents that Contain Your the Phrase: {}'.format(keyphrase))
    s = [i for i in nonfatal_words.keys() if i not in fatal_words and keyphrase in i]
    s.sort()
    for a in s:
        print(a)

    print('\n\n\n','---------------------------','\n\n\n')