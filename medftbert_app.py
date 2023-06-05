import streamlit as st
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

@st.cache_resource
def get_model():

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModelForTokenClassification.from_pretrained("jang007/Bert_finetuned_bc5cdr")
    return tokenizer,model



tokenizer, model = get_model()

user_input1 = st.text_area('Enter the medical transcript to analyze')
button = st.button('Analyze')


if user_input1 and button:
    label_list = ["O", "B-Chemical", "B-Disease", "I-Disease","I-Chemical"]
    test_sample = tokenizer([user_input1],padding = True, truncation = True, return_tensors='pt')
    # test_sample
    tokens = test_sample.tokens()
    outputs = model(**test_sample)
    predictions = torch.argmax(outputs.logits, dim=-1)
    predictions = predictions.cpu().detach().numpy()
    
    dise_list = []
    chem_list = []
    dise_word =""
    chem_word = ""

    for token, prediction in zip(tokens, predictions[0]):
        if label_list[prediction] != 'O':
            if token.startswith("##"):
                if label_list[prediction].endswith("Disease"):
                    dise_word = dise_word + token[2:]
                if label_list[prediction].endswith("Chemical"):
                    chem_word = chem_word + token[2:]
            else:
                if label_list[prediction].endswith("Disease"):
                    if len(dise_word) == 0:
                        dise_word = dise_word + token
                    else:
                        dise_word = dise_word + " " + token

                    if len(chem_word) != 0:
                        chem_list.append(chem_word)
                        chem_word=""
                if label_list[prediction].endswith("Chemical"):
                    if len(chem_word) == 0:
                        chem_word = chem_word + token
                    else:
                        chem_word = chem_word + " " + token
                    if len(dise_word) != 0:
                        dise_list.append(dise_word)
                        dise_word=""
        else:
            if len(chem_word) != 0:
                chem_list.append(chem_word)
                chem_word=""
            if len(dise_word) != 0:
                dise_list.append(dise_word)
                dise_word=""

    dise_nodup = []
    chem_nodup = []

    tmp= [dise_nodup.append(x) for x in dise_list if x not in dise_nodup]
    tmp= [chem_nodup.append(x) for x in chem_list if x not in chem_nodup]

    st.write("Diseases : ",str(dise_nodup))
    st.write("Medicines: ",str(chem_nodup))




# pip install -r .\requirements.txt
# streamlit run .\medftbert_app.py

# git init
# git add .\medftbert_app.py .\requirements.txt
# git commit -m 'first commit'
# git branch -M main
# git remote add origin https://github.com/jangg007/med_finetune_bert.git
# git push -u origin main