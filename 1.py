from transformers import AutoTokenizer
from transformers import AutoModel
from tqdm.auto import tqdm
from torch import nn
checkpoint='bert-base-chinese'
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
aa=input()
print(tokenizer(aa))