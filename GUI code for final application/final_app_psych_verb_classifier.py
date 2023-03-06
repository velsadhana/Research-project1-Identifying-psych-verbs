import tkinter as tk
from tkinter import filedialog as fd
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import numpy as np
import os
import random

os.environ['PYTHONHASHSEED']='0'
random.seed(6)
np.random.seed(6)
tf.random.set_seed(6)

# for creating GUI window
win = tk.Tk()

# adding title to the window
win_title = win.title("Psych verbs identifier")

# uploading the model 
def upld_model():
    path = fd.askopenfilename()
    path_txtbx3.insert(0,path)

# adding frame for the window
frame1 = tk.Frame(master=win, bg="#CCD1D1")
frame1.pack(fill=tk.BOTH, expand=True)

# label1 'Enter the sentence'
lbl1=tk.Label(master=frame1, text="Enter the sentence",bg="#CCD1D1",fg="black",font=('Lucida Calligraphy',11,'bold'))
lbl1.grid(row=1, column=1, padx=10, pady=10)

# textbox for displaying the sentence
path_txtbx1 = tk.Entry(master=frame1, font=('aerial',11),bg="#F2F3F4")
path_txtbx1.grid(row=1, column=2, padx=10, pady=10)

# label2 'Enter verb position'
lbl2=tk.Label(master=frame1, text="Enter verb position",bg="#CCD1D1",fg="black",font=('Lucida Calligraphy',11,'bold'))
lbl2.grid(row=2, column=1, padx=10, pady=10)

# textbox for displaying the verb position
path_txtbx2 = tk.Entry(master=frame1, font=('aerial',11),bg="#F2F3F4")
path_txtbx2.grid(row=2, column=2, padx=10, pady=10)


# button for uploading the model
upld_btn3 = tk.Button(
    text="Upload model",
    width=20,
    bg="#5DADE2",
    fg="black",
    font=('Lucida Calligraphy',11,'bold'),
    master=frame1,
    command=upld_model
    )
upld_btn3.grid(row=3, column=1, padx=10, pady=10)

# textbox for displaying the model path 
path_txtbx3 = tk.Entry(master=frame1,font=('aerial',11),bg="#F2F3F4")
path_txtbx3.grid(row=3, column=2, padx=10, pady=10)

# big text box for displaying the result
acc_txt_bx = tk.Entry(master=frame1,font=('calibri',20),bg="#F2F3F4",fg="blue")
acc_txt_bx.grid(row=2, column=3, padx=10, pady=10)


# tokenize the sentence
def tokenize(sentence):

    # using "bert-base-german-cased" model, since the dataset is German
    tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased")   

    tkens = tokenizer(sentence)
    token_words= tokenizer.convert_ids_to_tokens(tkens["input_ids"])

    # converting the input_ids to numpy arrays inorder to pass it into bert layer.
    inputids= np.array([tkens["input_ids"]])
    return [inputids,token_words]

# Getting the verb data such as verbs, verb count, previous & next words of a verb
def get_verb_data(sentence, token_id):

    # list to store no. of occurences of a verb in each sentence
    vrb_cnt = []
    # list to store the previous & next words of a verb when it occurs more than once in a sentence
    pn_lst=[]
    # list to store the previous & next words of a verb when it occurs more 
    # than once as well as in first position in a sentence
    pn1_lst=[]

    # finding the verbs whose position = tokenID-1 and storing it in list
    wrds_lst = sentence.split()
    vrb=wrds_lst[token_id-1]

    # finding the verb count
    vrb_cnt= int(wrds_lst.count(vrb))

    # finding previous and next words of a verb when it's count is >1
    if vrb_cnt>1 and token_id-1>0:
        pn_lst=[wrds_lst[token_id-2],wrds_lst[token_id]]
        
    if vrb_cnt>1>1 and token_id-1==0:
        pn1_lst=["[CLS]",wrds_lst[token_id]]
    print("verb",vrb) 
    return [vrb,vrb_cnt,pn_lst,pn1_lst]  

# Finding the position of verbs.
# For each sentence, finding the position of sub-words of a verb which we need.
def find_verb_pos(token_words,verb):
    # lists for storing the sub-words of verbs and their positions.
    sub_words=[]
    pos=[]

    # enumerate is used to count the index of words in a list. It is mainly used to find the index of 
    # duplicate words in a list.
    sub_words = [wrd for loc,wrd in enumerate(token_words) if wrd.replace('##', '') in verb]
    pos =[loc for loc,wrd in enumerate(token_words) if wrd.replace('##', '') in verb]
           
    # selecting only the sequenced numbers from the 'pos' list

    if verb in sub_words:
            seq_position=[y for x,y in zip(sub_words,pos) if x==verb]
    else:
            seq_position=[]
            p=0
            for x,y in zip(sub_words,pos):
                 p=p+1
                 if x in verb:
                    l=pos[p-1:]
                    for t in zip(l, l[1:]):
                        if t[0]+1 == t[1]:
                            seq_position=list(set(seq_position+list(t)))
                            seq_position.sort()
                    break

    return seq_position

# Finding the verb which we need, if the same verb occurs more than once.
def find_exact_verb(pn_list,verb_count,seq_pos,token_words):
    seqpos_len=int(len(seq_pos))
    splts= int(seqpos_len / verb_count)
    x=0
    y= splts

    # finding which verb is our needed verb based on matching the next and previous words
    if token_words[seq_pos[x]-1].replace('##', '')in pn_list[0] and token_words[seq_pos[y-1]+1].replace('##', '') in pn_list[1]:
        seq_pos[x:y]
    else:
        for cnt in range(2,verb_count+1):
            x=y
            y+=splts
            if token_words[seq_pos[x]-1].replace('##', '') in pn_list[0] and token_words[seq_pos[y-1]+1].replace('##', '') in pn_list[1]:
                seq_pos[x:y]
                break
    return seq_pos[x:y]


# Getting the position of verbs which we need, if the same verb occurs more than once
def get_verb_pos(verb_count,token_id,seq_pos,token_words,pn_list,pn1_list):
    actual_pos=[]
    val=0
    val1=0
    if verb_count>1 and token_id-1 > 0:
            actual_pos=find_exact_verb(pn_list[val],verb_count,seq_pos,token_words)
    elif verb_count>1 and token_id-1 == 0:
            actual_pos=find_exact_verb(pn1_list[val1],verb_count,seq_pos,token_words)
    elif len(seq_pos) > 1:
            temp=[]
            for t in zip(seq_pos, seq_pos[1:]):
                if t[0]+1 == t[1]:
                    temp=list(set(temp+list(t)))
                    temp.sort()
                else: 
                    break
            actual_pos=temp
    else:
            actual_pos=seq_pos
    
    return actual_pos

# Getting Bert embeddings
def get_bert(bert_input_ids):
  bertModel = TFBertModel.from_pretrained("bert-base-german-cased")
  bert_output = bertModel(bert_input_ids)
  l_h_s=bert_output[0]
  return l_h_s

# Getting the embeddings of sentence and verb from Bert model
def get_sent_and_verb_embed(bert_embedding,verb_pos):
    cls = bert_embedding[:,0,:]
    temp=[cls]
    temp1 = []
    for p in verb_pos:
      x=bert_embedding[:,p,:]
      temp.append(x)
    temp_tensors=tf.stack(temp,axis=1)
    paddings = tf.constant([[0, 0,], [0, 6-temp_tensors.shape[1]], [0, 0]])
    tnsr = tf.pad(temp_tensors, paddings, 'CONSTANT', constant_values=0)
    print("tensor shape",tnsr.shape)
    print("Sentence and verb embedding",tnsr)
    return tnsr

def get_embed(sentence,token_id):
    tokens=tokenize(sentence)
    verb_data=get_verb_data(sentence,token_id)
    seq_pos = find_verb_pos(tokens[1],verb_data[0])
    verb_pos = get_verb_pos(verb_data[1],token_id,seq_pos,tokens[1],verb_data[2],verb_data[3])
    bert_embed=get_bert(tokens[0])
    snt_vrb_emd=get_sent_and_verb_embed(bert_embed,verb_pos)
    return snt_vrb_emd


# processing the model
def get_model():
    psych_model = tf.keras.models.load_model(path_txtbx3.get(),compile=False)
    sent_verb_embed=get_embed(path_txtbx1.get(),int(path_txtbx2.get()))
    mdl_prd = psych_model.predict(sent_verb_embed)
    if mdl_prd < 0.5:
         predict_labels="Non-psych verb"
    else:
         predict_labels="Psych verb"
    acc_txt_bx.delete(0,tk.END)
    acc_txt_bx.insert(0,predict_labels)

# button for predicting the verb
upld_btn4 = tk.Button(
    text="Predict verb",
    width=20,
    bg="#5DADE2",
    fg="black",
    font=('Lucida Calligraphy',11,'bold'),
    master=frame1,
    command=get_model
    )
upld_btn4.grid(row=1, column=3, padx=10, pady=10)

lbl3=tk.Label(master=frame1, text="Psych verbs identifier",bg="#CCD1D1",fg="blue",font=('Calibri',11,'bold'))
lbl3.grid(row=0, column=0, padx=10, pady=10)

win.mainloop()
