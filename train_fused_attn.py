import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import time
import random
import torch
import torch.nn as nn
import numpy as np
import argparse
from dataloader import normalize,denormalize,dataloader,test_dataloader
from model import FusedAttentionModel,FusedAttentionTransformerConfig


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="processed_data_96_336")
parser.add_argument("--output_time_steps", type=int, default=336,choices=[96,336],help="输出序列长度")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=int, default=3e-4)
parser.add_argument("--weight_decay", type=int, default=0.1)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--hidden_size", type=int, default=256)
parser.add_argument("--intermediate_size", type=int, default=768)
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--eval_interval", type=int, default=10)
parser.add_argument("--early_stop", type=int, default=20,help="连续n次eval_loss不下降时停止训练")
parser.add_argument("--max_epoch", type=int, default=10)
parser.add_argument("--output_dir", type=str, default="transformers_output")
parser.add_argument("--seed", type=int, default=43)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

args.output_dir=args.output_dir+"/"+str(args.output_time_steps)+"_LR="+str(args.lr)+"_WD="+str(args.weight_decay)+"_SEED"+str(args.seed)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


train_dataloader,mean,std=dataloader(args.data_path,"train",args.batch_size)
eval_dataloader,_,_=dataloader(args.data_path,"val",args.batch_size)
num_heads=args.hidden_size//64
config=FusedAttentionTransformerConfig(args.hidden_size,num_heads,args.dropout,args.intermediate_size,96,args.output_time_steps,7,args.num_layers)
model=FusedAttentionModel(config).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def eval(model,eval_dataloader):
    total_loss=0
    total_steps=0
    for step,batch in enumerate(eval_dataloader):
        total_steps+=1
        model.eval()
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = True
        with torch.no_grad():
            input = torch.tensor(batch["input_ids"]).cuda()
            labels = torch.tensor(batch["labels"]).cuda()
            loss, output = model(input, labels)
            total_loss+=loss.cpu().data.item()
    return total_loss/total_steps

def test(model,test_dataloader):
    total_mse = 0
    total_mae = 0
    total_steps = 0
    for step, batch in enumerate(test_dataloader):
        total_steps += 1
        model.eval()
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = True
        with torch.no_grad():
            input = torch.tensor(batch["input_ids"]).cuda()
            labels = torch.tensor(batch["labels"]).cuda()
            loss, output = model(input, labels)
            total_mse += loss.cpu().data.item()
            mae=nn.L1Loss()
            mae_loss=mae(output,labels)
            total_mae+=mae_loss.cpu().data.item()
    mse_loss=total_mse / total_steps
    mae_loss=total_mae / total_steps
    return mse_loss,mae_loss

#训练
global_step=0
best_eval_loss=100
early_stop_count=0
epoch=0
while early_stop_count<args.early_stop:
    epoch+=1
    if epoch > args.max_epoch:
        break
    for step,batch in enumerate(train_dataloader):
        global_step+=1
        model.train()
        input=torch.tensor(batch["input_ids"]).cuda()
        labels=torch.tensor(batch["labels"]).cuda()
        loss, output  = model(input,labels)
        loss.backward()
        optimizer.step()
        model.zero_grad()
        print("step:{} loss:{}".format(global_step,loss.cpu().data.item()))
        if global_step % args.eval_interval == 0:
            eval_loss = eval(model, eval_dataloader)
            print("global_step:{} eval_loss:{}".format(global_step, eval_loss))
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                early_stop_count = 0
                torch.save(model.state_dict(), args.output_dir + "/best_ckpt.bin")
            else:
                early_stop_count += 1
                if early_stop_count >= args.early_stop:
                    break

print("best_eval_loss:{}".format(best_eval_loss))

#测试
config=FusedAttentionTransformerConfig(args.hidden_size,num_heads,args.dropout,args.intermediate_size,96,args.output_time_steps,7,args.num_layers)
model=FusedAttentionModel(config).cuda()
model.load_state_dict(torch.load(args.output_dir+"/best_ckpt.bin"))
test_dataloader, _, _ = test_dataloader(args.data_path, "test", args.batch_size)
mse_loss,mae_loss=test(model,test_dataloader)
print("MSE Loss:{},MAE Loss:{} on test dataset".format(mse_loss,mae_loss))
