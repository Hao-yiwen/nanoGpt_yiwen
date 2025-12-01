# Bigram Language Model - 训练循环
# 从 train.ipynb 迁移而来

import torch
import torch.nn as nn
from torch.nn import functional as F

# =====================
# 参数都集中放在这里
# =====================
DATA_PATH = 'input.txt'
RANDOM_SEED = 1337
BATCH_SIZE = 32
BLOCK_SIZE = 8
LEARNING_RATE = 1e-3
TRAIN_ITERS = 1000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INITIAL_GENERATE_TOKENS = 100
FINAL_GENERATE_TOKENS = 300
EVAL_ITERS = 200
N_EMBED = 32

# # 下载数据（如果需要，可以取消注释）
# import subprocess
# subprocess.run(['wget', 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'])

# =====================
# 读取数据
# =====================
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters:", len(text))
print(text[:1000])

# 创建字符集和词汇表
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# 创建编码解码映射
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encodec = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encodec("hello"))
print(decode(encodec("hello")))

# 数据转为张量
data = torch.tensor(encodec(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

# 分割数据集
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]

# 设置随机种子
torch.manual_seed(RANDOM_SEED)

# 获取批次函数
def get_batch(split):
    data_split = train_data if split == 'train' else test_data
    ix = torch.randint(len(data_split) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data_split[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data_split[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
    
    
# Bigram 语言模型定义
class BigramLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embeding_table = nn.Embedding(vocab_size, N_EMBED)
        self.psotion_embeding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        token_embeds = self.token_embeding_table(idx) # (B, T, C)
        pos_embeds = self.psotion_embeding_table(torch.arange(T, dtype=torch.long, device=DEVICE)) # (T, C)
        x = token_embeds + pos_embeds # (B, T, C)
        logits = self.lm_head(token_embeds) # (B, T, vocab_size)
        
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            B, T = idx.shape
            logits = logits.view(B, T, -1)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# 创建模型
model = BigramLanguageModel().to(DEVICE)

# 测试初始 loss
xb, yb = get_batch('train')
logits, loss = model(xb, yb)
print(f"Initial loss: {loss}")
print(f"Logits shape: {logits.shape}")

# 训练前文本生成
print("\n=== 训练前生成的文本 ===")
print(decode(model.generate(idx=torch.zeros((1, 1), dtype=torch.long).to(DEVICE), max_new_tokens=INITIAL_GENERATE_TOKENS)[0].tolist()))

# 创建优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 训练循环
print("\n=== 开始训练 ===")
for iter_num in range(TRAIN_ITERS):
    
    if iter_num % EVAL_ITERS == 0:
        losses = estimate_loss()
        print(f"iter {iter_num}: train loss={losses['train']:.4f}, test loss={losses['test']:.4f}")
    
    xb, yb = get_batch('train')
    optimizer.zero_grad(set_to_none=True)
    logits, loss = model(xb, yb)
    loss.backward()
    optimizer.step()
    
    if iter_num % 100 == 0:
        print(f"iter {iter_num}: loss={loss.item()}")

print(f"\n训练完成！最终 loss: {loss.item()}")

# 训练后文本生成
print("\n=== 训练后生成的文本 ===")
print(decode(model.generate(idx=torch.zeros((1, 1), dtype=torch.long).to(DEVICE), max_new_tokens=FINAL_GENERATE_TOKENS)[0].tolist()))
