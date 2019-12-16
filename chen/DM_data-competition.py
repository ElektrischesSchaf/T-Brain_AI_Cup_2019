#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pickle, json, re, time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from tqdm import tqdm_notebook as tqdm
#from tqdm import tqdm
from tqdm import trange

from gensim.parsing import remove_stopwords


import os


# In[ ]:


dataset = pd.read_csv('../Sample_Code/task1/data/task1_trainset.csv', dtype=str)

### Remove (current) redundant columns.
dataset.drop('Title',axis=1,inplace=True)
dataset.drop('Categories',axis=1,inplace=True)
dataset.drop('Created Date',axis=1, inplace=True)
dataset.drop('Authors',axis=1,inplace=True)
dataset['Abstract'] = dataset['Abstract'].str.lower()
#dataset['Task 1'] = dataset['Task 1'].str.lower()

for i in range(len(dataset['Abstract'])):
    dataset['Abstract'][i] = remove_stopwords(dataset['Abstract'][i])

# set test_size=0.1 for validation split
trainset, validset = train_test_split(dataset, test_size=0.1, random_state=42)

trainset.to_csv('../Sample_Code/task1/data/trainset.csv', index=False)
validset.to_csv('../Sample_Code/task1/data/validset.csv', index=False)


# In[ ]:


### Remove (current) redundant columns of the test set.

dataset = pd.read_csv('../Sample_Code/task1/data/task1_public_testset.csv', dtype=str)
dataset.drop('Title',axis=1,inplace=True)
dataset.drop('Categories',axis=1,inplace=True)
dataset.drop('Created Date',axis=1, inplace=True)
dataset.drop('Authors',axis=1,inplace=True)
dataset['Abstract'] = dataset['Abstract'].str.lower()

for i in range(len(dataset['Abstract'])):
    dataset['Abstract'][i] = remove_stopwords(dataset['Abstract'][i])

dataset.to_csv('../Sample_Code/task1/data/testset.csv', index=False)


# In[ ]:





# In[ ]:


dataset = pd.read_csv('../Sample_Code/task1/data/trainset.csv', dtype=str)


# In[ ]:


dataset.head()


# In[ ]:


sent_list = []
label_list = []
for i in dataset.iterrows():
    # remove $$$ and append to sent_list
    
    sent_list += i[1]['Abstract'].split('$$$')
    label_list += i[1]['Task 1'].split(' ')


# In[ ]:


df = pd.DataFrame({'Abstract': sent_list,
                   'Label': label_list})


# In[ ]:


def label_to_onehot(labels):
    """ Convert label to onehot .
        Args:
            labels (string): sentence's labels.
        Return:
            outputs (onehot list): sentence's onehot label.
    """
    label_dict = {'BACKGROUND': 0, 'OBJECTIVES':1, 'METHODS':2, 'RESULTS':3, 'CONCLUSIONS':4, 'OTHERS':5}
    onehot = [0,0,0,0,0,0]
    for l in labels.split('/'):
        onehot[label_dict[l]] = 1
    return tuple(onehot)

df['Onehot'] = df['Label'].apply(label_to_onehot)


# In[ ]:


df = df.loc[:, ['Abstract', 'Onehot']]


# In[ ]:


df.rename(columns={'Abstract': 0, 'Onehot': 1})


# In[ ]:


# set test_size=0.1 for validation split
trainset, validset = train_test_split(df, test_size=0.1, random_state=42)


# In[ ]:


get_ipython().system('pip install simpletransformers')


# In[ ]:


from simpletransformers.classification import MultiLabelClassificationModel


# In[ ]:


model = MultiLabelClassificationModel('roberta', 
                                      'roberta-base',
                                      num_labels=6, 
                                      args={'output_dir': 'outputs/',
                                            'max_seq_length': 128,
                                            'train_batch_size': 8,
                                            'eval_batch_size': 8,
                                            'num_train_epochs': 20,
                                            'learning_rate': 4e-5,
                                            'save_steps': 2000,
                                            'reprocess_input_data': True, 
                                            'overwrite_output_dir': True})


# In[ ]:


model.train_model(trainset)


# In[ ]:


result, model_outputs, wrong_predictions = model.eval_model(validset)


# In[ ]:


model_outputs


# In[ ]:


result


# In[ ]:





# In[ ]:





# In[ ]:


model = MultiLabelClassificationModel('roberta', 'outputs/')


# In[ ]:





# In[ ]:


testset = pd.read_csv('../Sample_Code/task1/data/testset.csv', dtype=str)


# In[ ]:


testset.tail()


# In[ ]:


sent_list = []
sid_list = []
limit = 0

for index, row in testset.iterrows():
    # remove $$$ and append to sent_list
    new_sent = row['Abstract'].split('$$$')
    sent_list += new_sent
    # Construct sid_list
    N = len(new_sent) + 1
    for i in range(1, N):
        sid = '%s_S%.3d' % (row['Id'], i)
        sid_list.append(sid)
    # limit = limit + 1
    # if limit > 100:
    #     break


# In[ ]:


len(sent_list), len(sid_list)


# In[ ]:


preds, outputs = model.predict(sent_list)


# In[ ]:


len(preds)


# In[ ]:


preds


# In[ ]:


outputs


# In[ ]:


submit_df = pd.DataFrame({'order_id': sid_list,
                           'BACKGROUND': None,
                           'OBJECTIVES': None,
                           'METHODS': None,
                           'RESULTS': None,
                           'CONCLUSIONS': None,
                           'OTHERS': None,
                           'preds': preds})


# In[ ]:


submit_df


# In[ ]:


submit_df['BACKGROUND'] = submit_df['preds'].apply(lambda x: x[0])
submit_df['OBJECTIVES'] = submit_df['preds'].apply(lambda x: x[1])
submit_df['METHODS'] = submit_df['preds'].apply(lambda x: x[2])
submit_df['RESULTS'] = submit_df['preds'].apply(lambda x: x[3])
submit_df['CONCLUSIONS'] = submit_df['preds'].apply(lambda x: x[4])
submit_df['OTHERS'] = submit_df['preds'].apply(lambda x: x[5])
submit_df.drop(['preds'], axis=1, inplace=True)


# In[ ]:


private_testset = pd.read_csv('../Sample_Code/task1/data/task1_sample_submission.csv')


# In[ ]:


private_testset = private_testset.iloc[131166:, :]


# In[ ]:


private_testset.head()


# In[ ]:


submit_df = pd.concat([submit_df, private_testset])


# In[ ]:


submit_df.tail()


# In[ ]:


submit_df.to_csv('submit2.csv', index=False)


# In[ ]:




