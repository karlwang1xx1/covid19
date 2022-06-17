import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime
from datetime import timedelta
''' for notebook in colab
from google.colab import files
files.upload()
'''   
import warnings
import pickle
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

#Importing the raw data

print('upload raw data from files')
try:
    encounters = pd.read_csv('raw_data/encounters.csv', usecols=['PATIENT','START','REASONCODE','CODE'], parse_dates =['START'])
    patients = pd.read_csv('raw_data/patients.csv',  usecols= ['Id','RACE','GENDER','BIRTHDATE','DEATHDATE'], parse_dates =['BIRTHDATE','DEATHDATE'])
except FileNotFound:
    print("File not found, please make sure encounters.csv and patients.csv in the raw_data/ folder")
# Find all patients that were admitted to the hospital because of Covid-19

filtered_index = np.where((encounters['REASONCODE'] == 840539006)  & ((encounters['CODE'] == 1505002) | (encounters['CODE'] == 305351004 )))
covid19 = encounters.loc[filtered_index]
# rename for merge
e_c = covid19.loc[:,('PATIENT','START')]  # beteter way to use covid19[['PATIENT','START']]
e_c.rename({'PATIENT': 'Id'}, axis='columns', inplace=True)
e_c['START'] = e_c['START'].dt.tz_localize(None)


#extract relevant patient info
p_c = patients.loc[:,('Id','RACE','GENDER','BIRTHDATE','DEATHDATE')]
p_c['DEATHDATE']= p_c['DEATHDATE'].fillna(0)
p_c['DEATHDATE'] = p_c['DEATHDATE'].where(p_c['DEATHDATE'] == 0, 1)
# label for COVID19 death
p_c.rename({'DEATHDATE': 'TOBE'}, axis='columns', inplace=True)


# merge two tables
inner_merged = pd.merge(e_c, p_c, on=["Id"])

# get the age and add as a column
inner_merged['START'] = pd.to_datetime(inner_merged['START']).dt.date
inner_merged['BIRTHDATE'] = pd.to_datetime(inner_merged['BIRTHDATE']).dt.date
inner_merged['AGE'] = (inner_merged['START']-inner_merged['BIRTHDATE']).dt.days/365.2425
#print(inner_merged)

# Make a dataset for modeling
dataset= inner_merged[['GENDER','RACE','AGE','TOBE']]
#print(dataset)
#print(dataset.dtypes)

#Here checked, no missing values, otherwise use sciikit imputater
#Use seaborn for initial visualizations
# Apply the default theme
sns.set_theme()
# Create a visualization
rel = sns.relplot(
    data=dataset,
    x="AGE", y="RACE", col="TOBE",
)
rel.set(title = "Pre feature checking: age is the major feature!")
plt.show()

dis = sns.displot(data=dataset, x="AGE", col="TOBE", kde=True)
dis.set(title = "The diagram of the major feature!")
dataset['TOBE'] = dataset['TOBE'].astype(str).astype(int)
mean_to_be = round(dataset.query('TOBE == 0')['AGE'].mean())
mean_not_to_be = round(dataset.query('TOBE == 1')['AGE'].mean())
plt.axvline(mean_to_be, color='green', label="mean age for ToBe:"+ str(mean_to_be) ) 
plt.axvline(mean_not_to_be, color='red', label="mean age for NotToBe:" + str(mean_not_to_be))
plt.legend(bbox_to_anchor=(-0.57, 0.999, 1, 0), loc=2, ncol=2, mode="expand", borderaxespad=0)
plt.savefig("Seaborn_displot_histogram_Python.png")
plt.show()

cat = sns.catplot(data=dataset, kind="violin", x="RACE", y="AGE", hue="TOBE", split=True)
cat.set(title = "Age distribution comparisons side bu side")
plt.show()

##########################
