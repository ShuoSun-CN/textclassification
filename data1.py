import torch
from datasets import load_dataset
from transformers import AutoTokenizer

class Dataset(torch.utils.data.Dataset):
    def __init__(self,data_type):
        self.data=self.load_data(data_type)

    def load_data(self,data_type):
        tmp_dataset=load_dataset(path='seamew/ChnSentiCorp',split=data_type)
        #获取huggingface数据
        Data={}
        for idx,line in enumerate(tmp_dataset):
            sample=line
            Data[idx]=sample
        return Data
    #处理数据
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
train_data=Dataset('train')
valid_data=Dataset('validation')
test_data=Dataset('test')
#将数据转变为torch可用类型

print(test_data[0])
from torch.utils.data import DataLoader

checkpoint='bert-base-chinese'
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
#将数据文字向量化处理



def collote_fn(batch_samples):
    batch_text=[]
    batch_label=[]
    for sample in batch_samples:
        batch_text.append(sample['text'])
        batch_label.append(int(sample['label']))
    X=tokenizer(
        batch_text,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    y=torch.tensor(batch_label)
    return X,y
#定义数据处理方式
train_dataloader=DataLoader(train_data,batch_size=4,shuffle=True,collate_fn=collote_fn)
valid_dataloader=DataLoader(valid_data,batch_size=4,shuffle=True,collate_fn=collote_fn)
test_dataloader=DataLoader(test_data,batch_size=4,shuffle=True,collate_fn=collote_fn)
#将数据加载
batch_X,batch_y=next(iter(train_dataloader))
print(train_dataloader.dataset)