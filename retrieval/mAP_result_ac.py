import sys 
sys.path.append(".") 
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import json

def get_similarity(f,features):
    scores = F.cosine_similarity(f,features,dim = -1)

    return scores

val_feature_file = "ckpt/activitynet_ckpt/validation_video_features.pkl"
train_feature_file = "ckpt/activitynet_ckpt/training_video_features.pkl"
annotation_file="datasets/activitynet/annotation.json"

is_average = True

with open(val_feature_file,"rb") as f:
    val_data = pickle.load(f)
with open(train_feature_file,"rb") as f:
    train_data = pickle.load(f)

with open(annotation_file,"r") as f:
    annotation_data = json.load(f)

label_idx = {
    label : i for i,label in enumerate(annotation_data['labels'])
}

# print(label_idx)

val_data = val_data
train_data = train_data
val_n = len(val_data)
train_n = len(train_data)

train_features = torch.zeros((train_n,512))
train_video_names = []
train_labels = []

val_features = torch.zeros((val_n,512))
val_video_names = []
val_labels = []

for idx,key in enumerate(train_data.keys()):
    label = annotation_data['database'][key]['label']
    train_features[idx] = torch.from_numpy(train_data[key])
    train_video_names.append(key)
    train_labels.append(label_idx[label])

for idx,key in enumerate(val_data.keys()):
    label = annotation_data['database'][key]['label']
    val_features[idx] = torch.from_numpy(val_data[key])
    val_video_names.append(key)
    val_labels.append(label_idx[label])



train_features = F.normalize(train_features,dim = -1)
val_features = F.normalize(val_features,dim = -1)

train_labels = np.array(train_labels)
val_labels = np.array(val_labels)

topk = 100
acc = 0.
total = val_n


aps = []

for idx,f in enumerate(val_features):
    similarity = get_similarity(f,train_features)
    _,sorted_idxs = torch.topk(similarity,k=topk)

    l = (train_labels[sorted_idxs] == val_labels[idx])


    total_correct = sum(l)

    precision_idxs = []
    accs = 0

    for correct_idx , is_same_cls in enumerate(l) :
        if is_same_cls:
            precision_idxs.append(correct_idx)
    
    # print(precision_idxs)

    ap = 0

    if total_correct == 0:
        continue
    for i_correct , k in enumerate(precision_idxs):
        ap = ap + (i_correct + 1) / (k + 1)


    # print(ap / total_correct)
    aps.append(ap / total_correct)

print("the mAP is {}".format(np.mean(aps)))


