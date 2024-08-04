'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

# import the necessary packages
import pandas as pd
import numpy as np


# Your code here

#Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
pred_universe_raw =pd.read_csv('../data/pred_universe_raw.csv')
arrest_events_raw =pd.read_csv('../data/arrest_events_raw.csv')

#Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
df_arrests = arrest_events_raw.merge(pred_universe_raw, how='outer', on='person_id')

#Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
df_arrests['diff'] = ( pd.to_datetime(df_arrests['arrest_date_event']) -pd.to_datetime(df_arrests['arrest_date_univ'])).dt.days

#Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise.
df_arrests['current_charge_felony'] = np.where(df_arrests['charge_degree'] == 'felony' ,1,0)

#df_arrests['y'] = np.select(([df_arrests['charge_degree'] == 'felony' ,  (df_arrests['diff']>0 &df_arrests['diff'] <= 365)]) ,1,0) 
                          #  & df_arrests[df_arrests.arrest_date_univ.between((df_arrests['arrest_date_event'] + pd.DateOffset(days=1)), (df_arrests['arrest_date_event'] + pd.DateOffset(years=1)))] ),1,0)
                           # & (df_arrests['arrest_date_univ'] > df_arrests['arrest_date_event'] + pd.Timedelta(days=1) 
                           #    & df_arrests['arrest_date_univ'] <= df_arrests['arrest_date_event'] + pd.Timedelta(years=1) )),1,0)
#df_arrests.loc[df_arrests['diff']>0 & ['diff'] <= 365, 'y' ] =1
'''this is an embarassing solution!'''
df_arrests['0'] = np.where(df_arrests['diff'] > 0 ,1,0)
df_arrests['-0'] = np.where(df_arrests['diff'] < 0 ,1,0)
df_arrests['365'] = np.where(df_arrests['diff'] <= 365 ,1,0)
df_arrests['-365'] = np.where(df_arrests['diff'] >= -365 ,1,0)
df_arrests['sum'] = df_arrests['current_charge_felony']+df_arrests['0']+df_arrests['365']
df_arrests['-sum'] = df_arrests['current_charge_felony']+df_arrests['-0']+df_arrests['-365']

df_arrests['y'] = np.where(df_arrests['sum'] ==3 ,1,0)
#print(df_arrests.head(20))
totalarrest = len(df_arrests)
#print(totalarrest)

df_rearrested = df_arrests.query('y == 1')
#print(df_rearrested.head())
rearrested = len(df_rearrested)
#print(rearrested)
ratio_rearrested = np.round((rearrested/totalarrest)*100,2)

print('\nWhat share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?')
print(f'out of {totalarrest} there were {rearrested} rearrested. {ratio_rearrested}% of the list.')

df_felony = df_arrests.query('current_charge_felony == 1')
#print(df_felony.head())
felony = len(df_felony)
#print(df_felony)
ratio_felony = np.round((felony/totalarrest)*100,2)

print('\nWhat share of current charges are felonies?')
print(f'out of {totalarrest} there were {felony} with felony charges. {ratio_felony}% of the list.')

df_arrests['num_fel_arrests_last_year'] =np.where (df_arrests['-sum'] ==3 ,1,0)
df_lastYearArrest = df_arrests.query('num_fel_arrests_last_year == 1')
#print(df_lastYearArrest.head())
lastyearArrest = len(df_lastYearArrest)
#print(lastyearArrest)
ratio_lastyearArrest = np.round((lastyearArrest/totalarrest)*100,2)
avg_lastyeararrest = df_arrests['num_fel_arrests_last_year'].mean()

print('\nWhat is the average number of felony arrests in the last year?')
print(f'out of {totalarrest} there were {lastyearArrest} arrested last prior year. {ratio_lastyearArrest}% of the list.')
print(f'average {avg_lastyeararrest}.\n')

df_arrests = df_arrests[['person_id','arrest_id_x','charge_degree','offense_category','arrest_date_event','arrest_id_y','age_at_arrest','sex','race','arrest_date_univ','y','current_charge_felony','num_fel_arrests_last_year']]
print(df_arrests.head())

def getArrestDF():
    return df_arrests