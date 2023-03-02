import tkinter as tk
from tkinter import filedialog as fd
import pandas as pd
import tensorflow as tf
import pickle

# for creating GUI window
win = tk.Tk()

# adding title to the window
win_title = win.title("Identifying psych verbs")

# uploading the test_df file
def upld_test_file():
    path = fd.askopenfilename()
    path_txtbx1.insert(0,path)

# uploading the test embeddings file
def upld_embed_file():
    path = fd.askopenfilename()
    path_txtbx2.insert(0,path)

# uploading the model 
def upld_model():
    path = fd.askopenfilename()
    path_txtbx3.insert(0,path)

# adding frame for the window
frame1 = tk.Frame(master=win, bg="#E0B1EC")
frame1.pack(fill=tk.BOTH, expand=True)

# button for uploading the test file
upld_btn1 = tk.Button(
    text="Upload test dataset",
    width=20,
    bg="#FBDEC8",
    fg="brown",
    font=('Lucida Calligraphy',11,'bold'),
    master=frame1,
    command=upld_test_file
    )
upld_btn1.grid(row=0, column=0, padx=10, pady=10)

# textbox for displaying the test file path 
path_txtbx1 = tk.Entry(master=frame1, font=('aerial',11),bg="#F2F3F4")
path_txtbx1.grid(row=0, column=1, padx=10, pady=10)

# button for uploading the embed file
upld_btn2 = tk.Button(
    text="Upload embedding file",
    width=20,
    bg="#FBDEC8",
    fg="brown",
    font=('Lucida Calligraphy',11,'bold'),
    master=frame1,
    command=upld_embed_file
    )
upld_btn2.grid(row=1, column=0, padx=10, pady=10)

# textbox for displaying the embed file path 
path_txtbx2 = tk.Entry(master=frame1,font=('aerial',11),bg="#F2F3F4")
path_txtbx2.grid(row=1, column=1, padx=10, pady=10)

# button for uploading the model
upld_btn3 = tk.Button(
    text="Upload model",
    width=20,
    bg="#FBDEC8",
    fg="brown",
    font=('Lucida Calligraphy',11,'bold'),
    master=frame1,
    command=upld_model
    )
upld_btn3.grid(row=2, column=0, padx=10, pady=10)

# textbox for displaying the model path 
path_txtbx3 = tk.Entry(master=frame1,font=('aerial',11),bg="#F2F3F4")
path_txtbx3.grid(row=2, column=1, padx=10, pady=10)

# processing the model
def get_model():
    psych_model = tf.keras.models.load_model(path_txtbx3.get(),compile=False)

    with open(path_txtbx2.get(), 'rb') as f:
        test_embeds = pickle.load(f)
    mdl_prd = psych_model.predict(test_embeds)
    return mdl_prd

# getting the predicted lables from the model
def get_test_df():
    prediction= get_model()
    predict_binary_labels = list(map(lambda x: 0 if x<0.5 else 1, prediction))
    predicted_labels = ["psych" if i==1 else "non-psych" for i in predict_binary_labels]

    # adding the predicted_label column to the test dataset file
    test_df=pd.read_excel(path_txtbx1.get())
    test_df["predicted_label"] = predicted_labels
    test_df["unmatched"] = ["matched" if i == j  else "unmatched" for i,j in zip(test_df["non-psych"],test_df["predicted_label"])]

    # calculating the test score
    final_score= str(len(test_df[test_df["unmatched"] == "matched"])/len(test_df["unmatched"]))

    # displaying score & predicted labels result
    final_display1="Final accuracy score: " + final_score+"\n\nTotal no. of data in test dataset: "+str(len(test_df))
    final_display2="\n\nNo. of data correctly predicted (actual label = predicted label): "+str(len(test_df[test_df["unmatched"] == "matched"]))
    final_display3="\n\nNo. of data not correctly predicted (actual label != predicted label): "+str(len(test_df[test_df["unmatched"] == "unmatched"]))
    
    # displaying verbs which are not correctly predicted by the model during testing
    temp_df=pd.DataFrame()
    temp_df = test_df.loc[test_df["unmatched"] == "unmatched"]
    temp_df = temp_df[["Verb"]]
    
    vrb_unmtch_dct = temp_df["Verb"].value_counts().to_dict()
    vrb_unmtch=str(vrb_unmtch_dct)
    final_display4 = "\n\nBelow are the verbs & their counts for which model doesn't predict correct label during testing "
    final_display=final_display1+final_display2+final_display3+final_display4+'\n\n'+vrb_unmtch

    return [test_df,final_display]

# generating the result in big text box
def get_display():
    test_output=get_test_df()
    acc_txt_bx.insert("1.0",test_output[1])

# saving the result file in local machine
def get_out_file():
    test_output=get_test_df()
    exl = fd.asksaveasfilename(defaultextension=".xlsx")
    test_output[0].to_excel(exl,index=False)

# button for predicting the verb
upld_btn4 = tk.Button(
    text="Predict verb",
    width=20,
    bg="#FBDEC8",
    fg="brown",
    font=('Lucida Calligraphy',11,'bold'),
    master=frame1,
    command=get_display
    )
upld_btn4.grid(row=0, column=6, padx=10, pady=10)

# big text box for displaying the result
acc_txt_bx = tk.Text(master=frame1,font=('calibri',14),bg="#F2F3F4",fg="blue")
acc_txt_bx.grid(row=2, column=6, padx=10, pady=10)

# label for asking whether to save the result file
lbl1=tk.Label(master=frame1, text="Do you want to save the result?",bg="#C963FF",fg="#FF0000",font=('Lucida Calligraphy',11,'bold'))
lbl1.grid(row=3, column=0, padx=10, pady=10)

# button 'Yes' for saving result file
yes_btn5 = tk.Button(
    text="Yes",
    bg="#FBDEC8",
    fg="brown",
    width=15,
    master=frame1,
    font=('Lucida Calligraphy',10,'bold'),
    command=get_out_file
    )
yes_btn5.grid(row=3, column=1, padx=10, pady=10)

# button 'No' for not to save result file & close the application
no_btn6 = tk.Button(
    text="No",
    width=15,
    bg="#FBDEC8",
    fg="brown",
    font=('Lucida Calligraphy',10,'bold'),
    master=frame1,
    command=win.destroy
    )
no_btn6.grid(row=3, column=2, padx=10, pady=10)

win.mainloop()
