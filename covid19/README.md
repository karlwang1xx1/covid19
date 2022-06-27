# To-be-or-Not-to-be-prediction
This GitHub repository is public

This project is to demonstrate model creation to predict a patient will survive or die after being hospitalized because of Covid-19.

The Jupyter Notebook file was included, named create_model.ipynb, which can be opened and run with the two original files in the same folder. 

A complete description for the project solution is in file `Note_Project.docx`.

To make the visualization portion work on Windows 10, it is important to have the correct version of matplotlib and python.
Python-3.9.0-amd64.exe, matplotlib-3.4.3-cp310-cp310-win_amd64.whl are included in this package for convenience if you are using Windows 10+.

On Windows, to setup the environment, the following steps as an example:
1.	create a virtual environment venv (python 3.9)
2.	venv\Scripts\activate 
3.	pip install -r requirements.txt

To run the program:
1.	python create.py to create models and save models in result_data folder
2.	python predict.py to load the saved models to predict survival or death
