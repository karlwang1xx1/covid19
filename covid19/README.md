# To-be-or-Not-to-be-prediction
This project is to show the model creation to predict the patient to either survive or die 
after hospitalized because of Covid19

A complete description for the project task solution is in file Note_Takehome_Project.docx

To make the visualization portion work, a matched pair of matplotlib and python is important regarding their versions
Python-3.9.0-amd64.exe, matplotlib-3.4.3-cp310-cp310-win_amd64.whl are included in this package for convenience 
if you are using wondows 10+

On Windows, when the enviroment is setup, the following steps as an example:
1) create a virtual enviroment venv (python 3.9)
2) venv\Scripts\activate 
3) pip install -requirements.txt
4) python create.py to creat models and save models in result_data (deep learning model in the create_model.ipynb, is not included in this version)
5) python predict.py to load the saved models to predit survive or die 

