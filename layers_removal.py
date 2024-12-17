#Code for removing layers from BERT
from transformers import TrainingArguments, Trainer
import numpy as np
import csv
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AlbertForSequenceClassification, AlbertConfig, AlbertTokenizer

from datasets import load_dataset, load_metric
import torch
from datasets import load_dataset, load_metric
import torch.nn as nn
import copy


#returns layers to keep
def inverse_layers(layers_to_remove):
    full_list=np.array(list(range(0,12)))
    ans=np.delete(full_list, np.array(layers_to_remove))
    return(list(ans))

def deleteEncodingLayers(model, layers_to_keep):  # must pass in the full finetuned model
    oldModuleList = model.bert.encoder.layer #'torch.nn.modules.container.ModuleList
    newModuleList = nn.ModuleList()
    # Now iterate over all layers, only keeping the relevant layers.
    for i in range(0, len(layers_to_keep)):
        newModuleList.append(oldModuleList[layers_to_keep[i]])
    # create a copy of the model, modify it with the new list, and return
    copyOfModel = copy.deepcopy(model)
    copyOfModel.bert.encoder.layer = newModuleList
    return copyOfModel

#substitute layers with new layers: for BERT model;
def substituteLayers(model, indx_layers_to_substitute, new_layers):
  copyOfModel=copy.deepcopy(model)
  copy_module_list=copyOfModel.bert.encoder.layer
  #change the layers in the copied model
  for i in range(0, len(indx_layers_to_substitute)):
    copy_module_list[indx_layers_to_substitute[i]]=new_layers[i]
  return copyOfModel

def extract_layers(model, indx_layers_to_extract):
  oldModuleList=model.bert.encoder.layer
  newModuleList = nn.ModuleList()
  for i in range(0, len(indx_layers_to_extract)):
    newModuleList.append(oldModuleList[indx_layers_to_extract[i]])
  return newModuleList

