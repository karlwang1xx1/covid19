import pickle
import sklearn
import pandas as pd
from tkinter import *
window =Tk()
window.title("Predict your patient to be (survive) or not to be (die)")
window.geometry('600x400')
window.wm_iconbitmap('concertai.ico') # Generals.ico is a filename for the window icon

lbl_error = Label(window, text='',fg="#f00")
lbl_error.grid(row=0, column=1, columnspan=3, sticky="w", padx=20)

lbl_g = Label(window, text='Gender')

lbl_r = Label(window, text='Race')

lbl_a = Label(window, text='Age')


lbl_g.grid(row=1, column=0, sticky="w", padx=20)
lbl_r.grid(row=1, column=1, sticky="w", padx=20)
lbl_a.grid(row=1, column=2, sticky="w", padx=20)




#gender = Entry(window) 
OPTIONS_G = [
"F",
"M",
] #etc
OPTIONS_R = [
"white",
"asian",
"black",
"native",
] #etc

def ok():
    try:
       float(age.get())
    except:
        print("please enter an integer for age")
        lbl_error["text"]="please enter an integer for age"
        return
    gender_ = v_g.get()
    race_ = v_r.get()
    age_ = float(age.get())
    print ("gender is:" + gender_)
    print ("  race is:" + race_)
    print ("   age is:" + str(age_))
    lbl_error["text"] =""
    # Transform
    X = pd.DataFrame(columns=["GENDER","RACE","AGE"],data=[[gender_,race_,age_]])
    #print(dataset)
    filename = 'result_data/ct_model.sav'
    ct = pickle.load(open(filename, 'rb'))
    cX = ct.transform(X)
    dX =pd.DataFrame(cX,columns=['encoder__GENDER_F', 'encoder__GENDER_M', 'encoder__RACE_asian',
       'encoder__RACE_black', 'encoder__RACE_native',
       'encoder__RACE_white', 'remainder__AGE'])
    #print(dX)
    filename = 'result_data/svm_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    p= loaded_model.predict(dX)
    print(p[0])
    color, message = pred(p[0])
    lbl_1["text"] = "Model 1: " + message
    lbl_1["fg"] = color

    filename = 'result_data/decision_tree_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    p= loaded_model.predict(dX)
    print(p[0])
    color, message = pred(p[0])
    lbl_2["text"] = "Model 2: " + message
    lbl_2["fg"] = color    

    filename = 'result_data/log_reg_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    p= loaded_model.predict(dX)
    print(p[0])
    color, message = pred(p[0])
    lbl_3["text"] = "Model 3: " + message
    lbl_3["fg"] = color

    filename = 'result_data/knn_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    p= loaded_model.predict(dX)
    print(p[0])
    color, message = pred(p[0])
    lbl_4["text"] = "Model 4: " + message
    lbl_4["fg"] = color
    return

def reset():    
    lbl_1["text"] =""
    lbl_2["text"] =""
    lbl_3["text"] =""
    lbl_4["text"] =""
    age.delete(0, END)

def pred(p):
    color ="#0f0"
    message =""
    if p==0:
        message ="To be"
    else:
        message = "Not to be"
        color ="#f00"
    return color, message

v_g = StringVar(window)
v_g.set(OPTIONS_G[0]) # default value

g = OptionMenu(window, v_g, *OPTIONS_G)
g.grid(row=2, column=0, sticky="w", padx=20)


v_r = StringVar(window)
v_r.set(OPTIONS_R[0]) # default value

r = OptionMenu(window, v_r, *OPTIONS_R)
r.grid(row=2, column=1, sticky="w", padx=20)

age = Entry(window)
age.grid(row=2, column=2, sticky="w", padx=20)
button = Button(window, text="Predict", command=ok)
button.grid(row=3,column=2, sticky="w", padx=20, pady=20)

btn_reset = Button(window, text="Reset", command=reset)
btn_reset.grid(row=3,column=2, sticky="w", padx=100, pady=20)

lbl_1 = Label(window, text='Model1:')
lbl_1.grid(row=4, column=1, columnspan=3, sticky="w", padx=20)
lbl_2 = Label(window, text='Model2:')
lbl_2.grid(row=5, column=1, columnspan=3, sticky="w", padx=20)
lbl_3 = Label(window, text='Model3:')
lbl_3.grid(row=6, column=1, columnspan=3, sticky="w", padx=20)
lbl_4 = Label(window, text='Model4:')
lbl_4.grid(row=7, column=1, columnspan=3, sticky="w", padx=20)

    
window.mainloop()

