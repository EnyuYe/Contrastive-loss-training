# from imports import Dataset,torch,np,random,DataLoader,util,tqdm,json
import numpy as np

from torch.utils.data import Dataset, DataLoader

from sentence_transformers import  util
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data.sampler import Sampler
# DATASET CLASSES PYTORCH 
# import helper_functions as hp
import torch
import itertools
import linecache
import random
def to_tensor(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x

class CustomDataset(Dataset):
    def __init__(self,dataset,edit_vectors_dict, neighbourhood_vectors_dict,paraphrase_vectors_dict,device):
        self.dataset=np.array(dataset,dtype=object)
        self.edit_vectors_dict=edit_vectors_dict
        self.neighbourhood_vectors_dict=neighbourhood_vectors_dict
        self.paraphrase_vectors_dict=paraphrase_vectors_dict
        self.device = device
 

    def __len__(self):
        return len(self.dataset)

    def total_indexes(self):
        return np.unique(self.dataset[:, 3])

    def get_row_indexes(self,target_sample_index):
        return np.where(self.dataset[:, 3] == target_sample_index)[0]

    def __getitem__(self, index):
        data_row=self.dataset[index]
        if(data_row[-2]==1):#paraphrase

            emb1 = to_tensor(self.edit_vectors_dict[data_row[1]]).to(self.device)#, dtype=torch.float)
            emb2 = to_tensor(self.paraphrase_vectors_dict[data_row[1]][data_row[0]]).to(self.device)#, dtype=torch.float)
            label = to_tensor(self.dataset[index][2]).to(self.device)#, dtype=torch.long) label
            sample_index=self.dataset[index][3]#sample index in dataset
            sent1=self.dataset[index][4]#sentence edit
            sent2=self.dataset[index][5]#paraphrase sentence
            pair_type=self.dataset[index][6]#neighbour,openai,paraphrase
            negative_sample_cntrl=self.dataset[index][7]
            emb1_index=data_row[0]#both should be the same
            emb2_index=data_row[1]#both should be the same

        else:#neighbour
            emb2 =to_tensor(self.edit_vectors_dict[data_row[1]]).to(self.device)#, dtype=torch.float)
            # print(data_row[0],data_row[0])
            emb1 = to_tensor(self.neighbourhood_vectors_dict[data_row[1]][data_row[0]]).to(self.device)#, dtype=torch.float)
            label = to_tensor(self.dataset[index][2]).to(self.device)#, dtype=torch.long)
            sample_index=self.dataset[index][3]
            sent1=self.dataset[index][4]
            sent2=self.dataset[index][5]
            pair_type=self.dataset[index][6]
            negative_sample_cntrl=self.dataset[index][7]
            emb1_index=data_row[0]
            emb2_index=data_row[1]
        
        return emb1, emb2, label, sample_index, sent1, sent2, pair_type, emb1_index, emb2_index, negative_sample_cntrl

def get_data_loader(dataset_paired, edit_vectors_dict, neighbourhood_train_vectors_dict, paraphrase_train_vectors_dict,batch_size=8192,shuffle=True,device="cpu"):
  """
    dataset: dataset to be used
    shuffle: dataset shuffle per iteration

  """

  dataset_pt=CustomDataset(dataset_paired, edit_vectors_dict, neighbourhood_train_vectors_dict, paraphrase_train_vectors_dict,device=device)
#   print(dataset_pt[0])
  data_loader=DataLoader(dataset_pt, batch_size=batch_size, shuffle=shuffle)
  return data_loader


def create_dataset_pairs(dataset,neightbour_control=0,label_reversal=False):
    """
    The dataloader is designed to create pairwise samples. 
    Since an edit can have multyple paraphrases storing edit vector multiple times is redundant thus
    all vectors are store in dictionary based on the sample index(order that sample appear in the dataset.

    dataset_paired_train/dataset_paired_test=
    [index1,index2,label,row_index,sentence1,sentence2,pair_type,control]
        index1 and index2:
            for edits index1 is always used and index1 == row_index since it is single value per sample
            for paraphrases, open_ai_paraphrases index1,index2 is used, this is not row_index, but index of the
                paraphrase in the list of paraphrases in a sample
            for neighbours index2 is used, this is not row_index, but index of the
                neighbour in the list of neighbours in a sample
            for open_ai_paraphrases index2 is used this is not row_index, but index of the
                open_ai_paraphrase in the list of open_ai_paraphrases in a sample
                
    Inputs:
    dataset: path to dataset
    neightbour_control=0# ignore
    label_reversal=False#ignore
    """
    paraphrase_label=1
    neightbour_label=0


    
 
    edit_vectors_dict={}#store all edit vectors
    neighbourhood_train_vectors_dict={}#store all training neigbours vectors
    neighbourhood_test_vectors_dict={}#store all test neighvours vectors
    paraphrase_train_vectors_dict={}#store all training paraphrases
    paraphrase_test_vectors_dict={}#store all test paraphrases

    dataset_paired_train=[]
    dataset_paired_test=[]
    for row_index,row in enumerate(dataset):
        
        edit_vectors_dict[row_index]=row["vector_edited_prompt"]#[0]
        
        check=True# used for setting an index that allows for manuipulating the dataset later on
        for index,vector in enumerate(row["vectors_paraphrases_train"]):
            if(row_index not in paraphrase_train_vectors_dict.keys()):
                paraphrase_train_vectors_dict[row_index]={}
            paraphrase_train_vectors_dict[row_index][index]=vector#[0]
            if(check):
                dataset_paired_train.append([index,row_index,paraphrase_label,row_index,
                                        row["edit_sentence"],row["paraphrases_train"][index],1,1])
                check=False
            else:
                dataset_paired_train.append([index,row_index,paraphrase_label,row_index,
                                        row["edit_sentence"],row["paraphrases_train"][index],1,0])

    
        for index,vector in enumerate(row["vectors_paraphrases_test"]):
            if(row_index not in paraphrase_test_vectors_dict.keys()):
                paraphrase_test_vectors_dict[row_index]={}
            paraphrase_test_vectors_dict[row_index][index]=vector#[0]
            dataset_paired_test.append([index,row_index,paraphrase_label,row_index,
                                    row["edit_sentence"],row["paraphrases_test"][index],1,0])
            break
        
        for index,vector in enumerate(row["vectors_neighbours_train"]):
            if(row_index not in neighbourhood_train_vectors_dict.keys()):
                neighbourhood_train_vectors_dict[row_index]={}
            neighbourhood_train_vectors_dict[row_index][index]=vector#[0]
            dataset_paired_train.append([index,row_index,neightbour_label,row_index,
                                    row["edit_sentence"],row["neighbours_train_question"][index],2,0])

            # print(vector[:5],row["neighborhood_prompts_high_sim"][index],"high")
        for index,vector in enumerate(row["vectors_neighbours_test"]):

            if(row_index not in neighbourhood_test_vectors_dict.keys()):
                
                neighbourhood_test_vectors_dict[row_index]={}
            neighbourhood_test_vectors_dict[row_index][index]=vector#[0]
            dataset_paired_test.append([index,row_index,neightbour_label,row_index,
                                    row["edit_sentence"],row["neighbours_train_question"][index],2,0])
    
    return  edit_vectors_dict, neighbourhood_train_vectors_dict, neighbourhood_test_vectors_dict, paraphrase_train_vectors_dict, paraphrase_test_vectors_dict, dataset_paired_train, dataset_paired_test


def data_construct_high_sim(edit_vectors_dict, neighbourhood_train_vectors_dict, paraphrase_train_vectors_dict, dataset_paired_train,neightbour_control=0,label_reversal=False,comparison="dist",topk_neg=0,pos_sim=0.65,loss="contrastive",device='cpu'):
    if(label_reversal==True):
        paraphrase=0
        neighbour=1
    else:
        paraphrase=1
        neighbour=0


    dataset_processed=[]

    vector_list_edits=[]
    vector_tensor_list_edits=[]
    edit_prompts=[]
    row_indexes_edits=[]
    data_index_edit=[]

    vector_list_neighbours=[]
    vector_tensor_list_neighbours=[]
    neighbours_prompt=[]
    neighbours_edits=[]
    data_index_edit_neighbours=[]

    data_loader=get_data_loader(dataset_paired_train, edit_vectors_dict, neighbourhood_train_vectors_dict,paraphrase_train_vectors_dict,batch_size=1,shuffle=False,device=device)
    
    for sample in data_loader:
        if(sample[2].item()==paraphrase):# only perfom for counterfact paraphrase not for openai
            if(sample[-1].item()!=1):#only add edit phrase once
                continue
            vector_list_edits.append(sample[0].detach().cpu().numpy().tolist())
            vector_tensor_list_edits.append(sample[0][0])
            edit_prompts.append(sample[4])
          
            row_indexes_edits.append(sample[3].item())
            data_index_edit.append(sample[3].item())
    
        else:
            vector_list_neighbours.append(sample[0].detach().cpu().numpy().tolist())
            vector_tensor_list_neighbours.append(sample[0][0])
            neighbours_prompt.append(sample[5])
            data_index_edit_neighbours.append(sample[-3].item())

    #THE COMMENTED OUT CODE IS TO MAKE PAIRS BETWEENS SIMILAR EDITS AND NEIGHBOURS ACROSS THE DATASET
    # print("topk_neg",topk_neg)
    # if(topk_neg!=0):
    #     vectors = torch.stack(vector_tensor_list_neighbours)
    #     for index_vector,(target_vector, edit_index) in enumerate(zip(vector_tensor_list_edits,data_index_edit)):# for each edit construct negative pairs across the dataset
    #         # print(target_vector,vectors[0])
    #         metric = util.cos_sim(target_vector,vectors)
    #         # print(metric)
    #         top_indices = torch.topk(metric, k=topk_neg).indices
    #         for index in top_indices[0].cpu().numpy().tolist():
    #         #   print(data_index_edit_neighbours[index])
    #             dataset_processed.append([data_index_edit_neighbours[index],edit_index,neighbour,row_indexes_edits[index_vector],
    #                                             edit_prompts[edit_index],neighbours_prompt[index],2,0])

                                 
    # print("done")  
   
    # for i, j in tqdm(itertools.combinations(range(len(vector_list_edits)), 2)):
    #     distance = util.cos_sim(vector_list_edits[i], vector_list_edits[j])
    #     if (distance > 0.80):
    #         dataset_processed.append([vector_list_edits[i],vector_list_edits[j],neighbour,row_indexes_edits[i],
    #                                     edit_prompts[i],edit_prompts[j],2,0])
    # print(type(vector_list_edits[0]))

    #PAIR EDITS WITH OTHER EDITS
    # The code works by creating a stack of edits vectors then computes pair wise cosine sim in the matrix
    # Self sim is set to zero and lower traingle of the matrix is considere cause upper and lower are the same.

    vectors_tensor = torch.stack(vector_tensor_list_edits)

    # Compute the magnitudes of each vector
    magnitudes = torch.norm(vectors_tensor, dim=1, keepdim=True)

    # Normalize each vector by dividing by its magnitude
    normalized_vectors = vectors_tensor / magnitudes
    # Compute pairwise cosine similarity matrix
    similarity_matrix = torch.matmul(normalized_vectors, normalized_vectors.t())

    # Fill diagonal with zeros to avoid self-similarity
    similarity_matrix.fill_diagonal_(0)

    # Find indices where similarity is greater than 0.80
    indices = torch.nonzero(similarity_matrix > pos_sim, as_tuple=False)

    indices = [[i.item(), j.item()] for i, j in indices if i < j]  # Ensure only one direction is considered

    # Iterate over the indices and update dataset_processed
    for i, j in tqdm(indices):
        if(data_index_edit[j] not in neighbourhood_train_vectors_dict[i].keys()):
            neighbourhood_train_vectors_dict[i][data_index_edit[j]]=vector_tensor_list_edits[j]#add to dict
            dataset_processed.append([
                data_index_edit[j],data_index_edit[i], neighbour, row_indexes_edits[i],
                edit_prompts[i][0], edit_prompts[j][0], 2,0])
    
    return dataset_processed,neighbourhood_train_vectors_dict

