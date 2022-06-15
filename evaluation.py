##Evaluation script

import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import values
from sklearn.metrics import f1_score,confusion_matrix,precision_score,recall_score,roc_auc_score,accuracy_score,average_precision_score
from sklearn.model_selection import train_test_split

##supressing_warnings
import warnings
warnings.filterwarnings("ignore")


def get_source_and_target_embedding(emb_file,sep,compression):
  """
  if a single embedding file is given with separating node id
  """

  if compression:
    data=pd.read_csv(emb_file,compression='gzip',header=None)
  else:
    data=pd.read_csv(emb_file)
  source_dict= dict((i,j) for i,j in enumerate(data[:sep].values))
  target_dict= dict((i+sep,j) for i,j in enumerate(data[sep:].values))

  return source_dict,target_dict


def get_negative_sampled_edges(data,source_node_range,target_node_range,
                                                      source_col,target_col
                                                                                ):

  """
  Function to return same number of negative samples as original data.
  We will fix the source and randomly sample a target node not in test data.

  """
  negative_samples=[]
  node_val=data[source_col].values

  np.random.seed(1345)
  for i in  tqdm(range(data.shape[0])):
    node=node_val[i]
    node_con=data[data[source_col]==node]
    rand_node=0
    target_val=list(node_con[target_col].unique())
    while(1) :

      if values.bipartite:
          rand_node=np.random.randint(target_node_range[0],target_node_range[1])
      else:
          rand_node=np.random.randint(source_node_range[0],target_node_range[1])
      
      if rand_node not in target_val :
        break
    negative_samples.append([node,rand_node])
  negative_samples=np.array(negative_samples)

  return negative_samples

def prepare_train_test_set_for_regression(data,source_node_range,target_node_range,
                                                      source_col,target_col,s_emb,t_emb,test_ratio):
  """
    This function prepares the train and test required for regression
  """
  neg_data=get_negative_sampled_edges(data,source_node_range,target_node_range,
                                                      source_col,target_col)                                                    
  pos_src_emb=np.array([ s_emb[x] for x in data[source_col].values])
  pos_tgt_emb=np.array([ t_emb[x] for x in data[target_col].values])
  
  neg_src_emb=np.array([ s_emb[x] for x in neg_data[:,0]])
  neg_tgt_emb=np.array([ {**s_emb,**t_emb}[x] for x in neg_data[:,1]])

  ##sample both the sets separately 

  pos_emb=np.concatenate((pos_src_emb,pos_tgt_emb),axis=1)
  neg_emb=np.concatenate((neg_src_emb,neg_tgt_emb),axis=1)

  train_pos_emb,test_pos_emb=train_test_split(pos_emb,test_size=test_ratio,shuffle=False)
  _,test_edge_id=train_test_split(data,test_size=test_ratio,shuffle=False)
  neg_data=pd.DataFrame(neg_data)
  neg_data.columns=[source_col,target_col]
  _,test_neg_edge_id=train_test_split(neg_data,test_size=test_ratio,shuffle=False)
  train_neg_emb,test_neg_emb=train_test_split(neg_emb,test_size=test_ratio,shuffle=False)

  train_x=np.concatenate((train_pos_emb, train_neg_emb),axis=0)
  train_y=np.concatenate((np.ones((train_pos_emb.shape[0],1)),np.zeros((train_neg_emb.shape[0],1))),axis=0)
  test_x=np.concatenate((test_pos_emb, test_neg_emb),axis=0)
  test_y=np.concatenate((np.ones((test_pos_emb.shape[0],1)),np.zeros((test_neg_emb.shape[0],1))),axis=0)

  return train_x,train_y,test_x,test_y,test_edge_id,test_neg_edge_id

def train_logistic_regression(train_x,train_y):
  model=LogisticRegression(max_iter=2000)
  model.fit(train_x,train_y)
  return model

def get_row_MRR(probs,true_classes):
        existing_mask = true_classes == 1
        #descending in probability
        ordered_indices = np.flip(probs.argsort())

        ordered_existing_mask = existing_mask[ordered_indices]
        existing_ranks = np.arange(1,
                                   true_classes.shape[0]+1,
                                   dtype=np.float)[[i for sl in ordered_existing_mask for i in sl]]

        MRR = (1/existing_ranks).sum()/existing_ranks.shape[0]
        return MRR


def evaluate(data,source_node_range,target_node_range,
                                                      source_col,target_col,s_emb,t_emb,test_ratio,train_data):
 

  ##get train and test data for regression
  train_x,train_y,test_x,test_y,test_edge_id,test_neg_edge_id=prepare_train_test_set_for_regression(data,source_node_range,target_node_range,
                                                      source_col,target_col,s_emb,t_emb,test_ratio)
  model=train_logistic_regression(train_x,train_y)

  prediction=model.predict(test_x)
  predicted_prob=model.predict_proba(test_x)
  predicted_prob=predicted_prob[:,1]

  print('-'*20+'Evaluation Report'+'-'*20+'\n')
  print(f'no of examples in train set {train_x.size} test set{test_x.size} for logistic regression \n')
  print('confusion matrix for test')
  print(confusion_matrix(test_y,prediction))
  print(f'\n\n precision {precision_score(test_y,prediction)} recall {recall_score(test_y,prediction)} \n f1_score {f1_score(test_y,prediction)} accuracy {accuracy_score(test_y,prediction) }')
  
  print(f' auc score {roc_auc_score(test_y,prediction)}')
  print('\n'+'-'*20+'other standard metrics'+'-'*20+ '\n ')
  print(f' MAP score {average_precision_score(test_y,predicted_prob)}')
  print(f'\n MRR   {get_row_MRR(predicted_prob,test_y)}')

  ##sorting the scores
  sorted_pred=[y for x,y in  sorted(zip(predicted_prob,prediction),reverse=True)]
  sorted_lab=[y for x,y in  sorted(zip(predicted_prob,test_y),reverse=True)]

  print(f'\n recall @10 {recall_score(sorted_lab[:10],sorted_pred[:10])}')
  print(f'\n precision @10 {precision_score(sorted_lab[:10],sorted_pred[:10])}')
  print(f'\n recall @100 {recall_score(sorted_lab[:100],sorted_pred[:100])}')
  print(f'\n precission @100 {precision_score(sorted_lab[:100],sorted_pred[:100])}')

  print('-'*50)

  prediction=prediction[:len(test_edge_id)]
  
  a=test_edge_id[[source_col,target_col]]
  b=train_data[[source_col,target_col]]
  ds1 = set(map(tuple, a.values))
  ds2 = set(map(tuple, b.values))
  
  ##addition_block
  ab=ds1.difference(ds2)

  ls=[]
  for k in a.values:
    if (k[0],k[1]) in list(ab):
      ls.append(True)
    else:
      ls.append(False)

  addition_pred=prediction[ls]

  print('*'*20+'\n addition_block \n'+'*'*20)
  print(f'No of targets {len(ls)} No of targets caught {addition_pred.sum()} percent= {addition_pred.sum()/len(ls)}')

  ab=ds1.intersection(ds2)

  ls=[]
  for k in a.values:
    if (k[0],k[1]) in list(ab):
      ls.append(True)
    else:
      ls.append(False)

  addition_pred=prediction[ls]
  
  print('*'*20+'\n retention_block \n'+'*'*20)
  print(f'No of targets {len(ls)} No of targets caught {addition_pred.sum()} percent= {addition_pred.sum()/len(ls)}')
  a=test_neg_edge_id[[source_col,target_col]]
  ds1 = set(map(tuple, a.values))
  ds2 = set(map(tuple, b.values))
  
  ab=ds1.intersection(ds2)

  ls=[]
  for k in a.values:
    if (k[0],k[1]) in list(ab):
      ls.append(True)
    else:
      ls.append(False)

  addition_pred=prediction[ls]
  print('*'*20+'\n deletion_block \n'+'*'*20)
  print(f'No of targets {len(ls)} No of targets caught {addition_pred.sum()} percent= {addition_pred.sum()/len(ls)}')
 

  



  

  # print()






if __name__=="__main__":
  if values.emb_file!="":
    s_emb,t_emb= get_source_and_target_embedding(values.emb_file,values.sep,values.compression)
  else:
    s_emb=json.load(values.source_emb_file)
    t_emb=json.load(values.target_emb_file)
  test_file=pd.read_csv(values.test_file)
  train_file=pd.read_csv(values.train_file)
  test_file=test_file.drop_duplicates()
  print('-'*50)
  print('\nHomogeneous prediction\n')
  print('-'*50)
  values.bipartite=0
  evaluate(test_file,values.source_node_range,values.target_node_range,values.source_col,
        values.target_col,s_emb,t_emb,values.test_ratio,train_file)
  print('-'*50)
  print('\nBipartite Prediction\n')
  print('-'*50)
  print('\n')
  values.bipartite=1
  evaluate(test_file,values.source_node_range,values.target_node_range,values.source_col,
        values.target_col,s_emb,t_emb,values.test_ratio,train_file)
  