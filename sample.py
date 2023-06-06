import torch.cuda

from transformers import AutoTokenizer
from torch import nn
from transformers import AutoModel
from tqdm.auto import tqdm
from torch import nn
checkpoint='bert-base-chinese'
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.bert_encoder = AutoModel.from_pretrained('bert-base-chinese')#包括一个基于中文文字的Encoder
        self.classifier = nn.Linear(768, 2)#和一个全连接层

    def forward(self, x):
        bert_output = self.bert_encoder(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0]
        logits = self.classifier(cls_vectors)
        return logits#使用logistics回归确定0和1的值
model=Net()
model.load_state_dict(torch.load('epoch_3_valid_acc_95.0_model_weights.bin'))
while True:
    X=input("请输入要判断的话:")
    X=[X]
    X = tokenizer(
        X,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    y=model(X)
    X=X['input_ids']


    y=y.tolist()


    y=y[0]
    if y[0]<=y[1]:
        print("这是一句好话。")
    else:
        print("这是一句坏话。")
