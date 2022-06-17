import os
import pickle
import sklearn
from tkinter import *
window =Tk()
window.title("Predict your patient to be or not to be")
window.geometry('400x400')
window.wm_iconbitmap('concertai.ico') # Generals.ico is a filename for the window icon
lbl_g = Label(window, text='Gender')

lbl_r = Label(window, text='Race')

lbl_a = Label(window, text='Age')


lbl_g.grid(row=1, column=0, sticky="w", padx=20)
lbl_r.grid(row=1, column=1, sticky="w", padx=20)
lbl_a.grid(row=1, column=2, sticky="w", padx=20)
lbl_error = Label(window, text='',fg="#f00")
lbl_error.grid(row=0, column=1, columnspan=3, sticky="w", padx=20)


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
    print ("gender is:" + v_g.get())
    print ("  race is:" + v_r.get())
    print ("   age is:" + age.get())
    lbl_error["text"] =""
    # Transform

    filename = 'result_data/svm_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    return
v_g = StringVar(window)
v_g.set(OPTIONS_G[0]) # default value

g = OptionMenu(window, v_g, *OPTIONS_G)
g.grid(row=1, column=0, sticky="w", padx=20)


v_r = StringVar(window)
v_r.set(OPTIONS_R[0]) # default value

r = OptionMenu(window, v_r, *OPTIONS_R)
r.grid(row=1, column=1, sticky="w", padx=20)

age = Entry(window)
age.grid(row=1, column=2, sticky="w", padx=20)
button = Button(window, text="Predict", command=ok)
button.grid(row=2,column=2, sticky="w", padx=20, pady=20)

lbl_1 = Label(window, text='Model1:')
lbl_1.grid(row=3, column=1, columnspan=3, sticky="w", padx=20)
lbl_2 = Label(window, text='Model2:')
lbl_2.grid(row=4, column=1, columnspan=3, sticky="w", padx=20)
lbl_3 = Label(window, text='Model3:')
lbl_3.grid(row=5, column=1, columnspan=3, sticky="w", padx=20)
lbl_4 = Label(window, text='Model4:')
lbl_4.grid(row=6, column=1, columnspan=3, sticky="w", padx=20)


window.mainloop()

