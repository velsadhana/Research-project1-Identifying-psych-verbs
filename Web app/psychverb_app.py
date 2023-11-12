
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import numpy as np
import os
import random
from flask import Flask, render_template, request,url_for, redirect

# setting the seeds
os.environ['PYTHONHASHSEED']='0'
random.seed(6)
np.random.seed(6)
tf.random.set_seed(6)

# Flask constructor 
app = Flask(__name__)   

# A decorator used to route url to display the html file
@app.route('/')       
def template(): 
    return render_template('design.html') 

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
    token_id = int(token_id)
    tokens=tokenize(sentence)
    verb_data=get_verb_data(sentence,token_id)
    seq_pos = find_verb_pos(tokens[1],verb_data[0])
    verb_pos = get_verb_pos(verb_data[1],token_id,seq_pos,tokens[1],verb_data[2],verb_data[3])
    bert_embed=get_bert(tokens[0])
    snt_vrb_emd=get_sent_and_verb_embed(bert_embed,verb_pos)
    return snt_vrb_emd


# submitting the input values when clicking submit button using post method
@app.route('/predict', methods=['POST'])
def predict():
    psych_model = tf.keras.models.load_model('C:\SadhanaOldlaptop\Velsadhana\Masterscourse\Linguistics Data Science\Sum sem 2022\Research project 1\Web app\model1.h5',compile=False)
    sent_verb_embed=get_embed(request.form['snt'],request.form['vp'])
    mdl_prd = psych_model.predict(sent_verb_embed)
    if mdl_prd < 0.5:
         predict_labels="Non-psych verb !"
         
    else:
         predict_labels="Psych verb !"
        
    return render_template('design.html', prediction_text= predict_labels)

# returning back to home page when clicking 'Try next sentence?' button
@app.route('/back', methods=['POST'])       
def back(): 
    return redirect(url_for('template'))

# directing back to verbs page when clicking 'verbs' link in the paragraph
@app.route('/verbs')       
def verbs(): 
    return render_template('verb.html')

if __name__=='__main__': 
   app.run(debug= True) 












