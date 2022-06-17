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

models =[] # will hold 4 model names
#Importing the raw data

print('upload raw data from files')
print('if a figre pops up, close it to move forward...')
try:
    encounters = pd.read_csv('raw_data/encounters.csv', usecols=['PATIENT','START','REASONCODE','CODE'], parse_dates =['START'])
except FileNotFoundError:
    print("File not found: raw_data/encounters.csv")
    exit()
except pd.errors.EmptyDataError:
    print("No data")
except pd.errors.ParserError:
    print("Parse error")
except Exception:
    print("Some other exception")
try:
    patients = pd.read_csv('raw_data/patients.csv',  usecols= ['Id','RACE','GENDER','BIRTHDATE','DEATHDATE'], parse_dates =['BIRTHDATE','DEATHDATE'])

except FileNotFoundError:
    print("File not found: raw_data/patients.csv")
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
plt.savefig("result_data/age_major_feature.png")
plt.show()

dis = sns.displot(data=dataset, x="AGE", col="TOBE", kde=True)
dis.set(title = "The diagram of the major feature!")
dataset['TOBE'] = dataset['TOBE'].astype(str).astype(int)
mean_to_be = round(dataset.query('TOBE == 0')['AGE'].mean())
mean_not_to_be = round(dataset.query('TOBE == 1')['AGE'].mean())
plt.axvline(mean_to_be, color='green', label="ToBe mean age:"+ str(mean_to_be) ) 
plt.axvline(mean_not_to_be, color='red', label="NotToBe mean age:" + str(mean_not_to_be))
plt.legend(bbox_to_anchor=(-0.57, 0.99, 1, 0), loc=2, ncol=2, mode="expand", borderaxespad=0)
plt.savefig("result_data/age_histogram.png")
plt.show()

cat = sns.catplot(data=dataset, kind="violin", x="RACE", y="AGE", hue="TOBE", split=True)
cat.set(title = "Age distribution comparisons side bu side")
plt.savefig("result_data/age_violin_plot.png")
plt.show()

########################## Modeling ####################
## Independent variables and dependent one
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Transform all categorical variables into numeric, usung One Hot Encoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['GENDER','RACE'])], remainder='passthrough')
cX = ct.fit_transform(X)
#print(cX)
## back to dataframe:
dX =pd.DataFrame(cX,columns=ct.get_feature_names_out())
##
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dX, y, test_size = 0.25, random_state = 0)

## Feature Scaling (skip this) optional as the numbers look good, not too big, tried but didn't do any better
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)
#X_train

## Model 1: Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
model = SVC(kernel = 'rbf', random_state = 0)
model.fit(X_train, y_train)
# save the model to disk
filename = 'result_data/svm_model.sav'
models.append(filename)
pickle.dump(model, open(filename, 'wb'))
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

## Model 2: Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 0)
model.fit(X_train, y_train)

# save the model to disk
filename = 'result_data/log_reg_model.sav'
models.append(filename)
pickle.dump(model, open(filename, 'wb'))
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

## Model 3: Training the Decision Tree model on the Training set
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
model.fit(X_train, y_train)

# save the model to disk
filename = 'result_data/decision_tree_model.sav'
models.append(filename)

pickle.dump(model, open(filename, 'wb'))
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

## Model 4: Training the K-Nearest Neighbor model on the Training set

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
model.fit(X_train, y_train)
# save the model to disk
filename = 'result_data/knn_model.sav'
models.append(filename)
pickle.dump(model, open(filename, 'wb'))
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

## Making the Confusion Matrix (Test the last model that was run from the above 4 models)

from sklearn.metrics import confusion_matrix, accuracy_score
for model in models:
    loaded_model = pickle.load(open(model, 'rb'))
    y_pred = loaded_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(model)
    print(cm)
    print (str(round(accuracy_score(y_test, y_pred)*100,2))+"%")
print(" 4 trained models are saved in result_data/ folder")









