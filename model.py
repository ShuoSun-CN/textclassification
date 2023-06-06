import torch.cuda

import data1
from torch import nn
from transformers import AutoModel
from tqdm.auto import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.bert_encoder = AutoModel.from_pretrained(data1.checkpoint)#包括一个基于中文文字的Encoder
        self.classifier = nn.Linear(768, 2)#和一个全连接层

    def forward(self, x):
        bert_output = self.bert_encoder(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0]
        logits = self.classifier(cls_vectors)
        return logits#使用logistics回归确定0和1的值


model = Net().to(device)


def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss:{0:>7f}')
    finish_batch_num = (epoch - 1) * len(dataloader)

    model.train()
    for batch, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss:{total_loss / (finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss


def test_loop(dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']
    size = len(dataloader.dataset)
    correct = 0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size
    print(f'{mode} Accuracy:{(100 * correct):>0.1f}%\n')
    return correct


from transformers import AdamW, get_scheduler

learning_rate = 1e-5
epoch_num = 3
loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num * len(data1.train_dataloader)
)
total_loss = 0
best_acc = 0
for i in range(epoch_num):
    print(f"epoch{i + 1}/{epoch_num}\n---------------")
    total_loss = train_loop(data1.train_dataloader, model, loss_fn, optimizer, lr_scheduler, i + 1, total_loss)
    valid_acc = test_loop(data1.valid_dataloader, model, mode="Valid")
    if valid_acc > best_acc:
        best_acc = valid_acc
        print('saving new weights...\n')
        torch.save(model.state_dict(), f'epoch_{i + 1}_valid_acc_{(100 * valid_acc):0.1f}_model_weights.bin')
print("Done")
