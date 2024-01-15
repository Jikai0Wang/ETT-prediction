import json
import numpy as np
import tqdm
#标准化
def normalize(data,mean,std):
    return (data - mean) / std
#反标准化
def denormalize(data,mean,std):
    return data*std+mean


def dataloader(datadir,split,batch_size):
    print("loading {} data".format(split))
    with open(datadir + "/meam_std.json", mode='r', encoding='utf-8') as f0:
        d = json.load(f0)
        mean=np.array(d["mean"])
        std=np.array(d["std"])
    with open(datadir+"/"+split+'.json', mode='r', encoding='utf-8') as f:
        dicts = json.load(f)
    batches=[]
    for i in tqdm.tqdm(range(len(dicts)//batch_size)):
        input_ids=[]
        labels=[]
        in_times=[]
        out_times=[]
        for j in range(batch_size):
            sample=dicts[batch_size*i+j]
            input_ids.append(normalize(np.array(sample["input"])[:,1:].astype(np.float64),mean,std).tolist())
            labels.append(normalize(np.array(sample["label"])[:,1:].astype(np.float64),mean,std).tolist())
            in_times.append(np.array(sample["input"])[:,0].tolist())
            out_times.append(np.array(sample["label"])[:, 0].tolist())
        batches.append({"input_ids":input_ids,"labels":labels,"in_times":in_times,"out_times":out_times})
    return batches,mean,std

def test_dataloader(datadir,split,batch_size):
    print("loading {} data".format(split))
    with open(datadir + "/meam_std.json", mode='r', encoding='utf-8') as f0:
        d = json.load(f0)
        mean=[]
        std=[]
        now_mean=[]
        now_std=[]
    with open(datadir+"/"+split+'.json', mode='r', encoding='utf-8') as f:
        dicts = json.load(f)
    batches=[]
    for i in tqdm.tqdm(range(len(dicts)//batch_size)):
        input_ids=[]
        labels=[]
        in_times=[]
        out_times=[]
        for j in range(batch_size):
            sample=dicts[batch_size*i+j]
            mean.append(np.mean(np.array(sample["input"])[:,1:].astype(np.float64),axis=0))
            std.append(np.std(np.array(sample["input"])[:,1:].astype(np.float64),axis=0))
            now_mean.append(np.mean(np.array(mean,dtype=np.float64),axis=0))
            now_std.append(np.mean(np.array(mean,dtype=np.float64),axis=0))
            input_ids.append(normalize(np.array(sample["input"])[:,1:].astype(np.float64),now_mean[-1],now_std[-1]).tolist())
            labels.append(normalize(np.array(sample["label"])[:,1:].astype(np.float64),now_mean[-1],now_std[-1]).tolist())
            in_times.append(np.array(sample["input"])[:,0].tolist())
            out_times.append(np.array(sample["label"])[:, 0].tolist())
        batches.append({"input_ids":input_ids,"labels":labels,"in_times":in_times,"out_times":out_times})
    return batches,mean,std
