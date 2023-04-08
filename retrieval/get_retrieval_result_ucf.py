import sys 
sys.path.append(".") 
import pickle
import torch
import torch.nn.functional as F


def get_similarity(f,features):
    scores = F.cosine_similarity(f,features,dim = -1)
    return scores

annotaion_file = "datasets/ucf_51/annotation.json"
val_feature_file = "ckpt/ucf51_ckpt/validation_video_features.pkl"
train_feature_file = "ckpt/ucf51_ckpt/training_video_features.pkl"
is_average = True

with open(val_feature_file,"rb") as f:
    val_data = pickle.load(f)
with open(train_feature_file,"rb") as f:
    train_data = pickle.load(f)



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
    label = key.split("_")[1]
    train_features[idx] = torch.from_numpy(train_data[key])
    train_video_names.append(key)
    train_labels.append(label)

for idx,key in enumerate(val_data.keys()):
    label = key.split("_")[1]
    val_features[idx] = torch.from_numpy(val_data[key])
    val_video_names.append(key)
    val_labels.append(label)

# train_features = F.normalize(train_features,dim = -1)
# val_features = F.normalize(val_features,dim = -1)


topk = 5
acc = 0.
total = val_n

for idx,f in enumerate(val_features):
    similarity = get_similarity(f,train_features)
    _,sorted_idxs = torch.topk(similarity,k=topk)

    for sorted_idx in sorted_idxs:
        if val_labels[idx] == train_labels[sorted_idx]:
            acc += 1
            break

print(acc)
print(total)
print("the top{} acc is {}".format(topk,acc / total))


