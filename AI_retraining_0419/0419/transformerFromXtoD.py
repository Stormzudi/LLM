import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

# 定义字符集
chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
vocab_size = len(chars)

# 定义字符到索引的映射
char2idx = {char: idx for idx, char in enumerate(chars)}
idx2char = {idx: char for idx, char in enumerate(chars)}

# 将字符转换为索引
def char2index(char):
    return char2idx[char]

# 将索引转换为字符
def index2char(index):
    return idx2char[index]

# 创建数据集
def create_dataset(num_samples):
    src_data = []
    tgt_data = []
    for _ in range(num_samples):
        # 随机生成5个小写字母
        src = ''.join(random.choices(chars[26:], k=5))
        # 将小写字母转换为大写字母
        tgt = src.upper()
        src_data.append([char2index(c) for c in src])
        tgt_data.append([char2index(c) for c in tgt])
    return torch.tensor(src_data), torch.tensor(tgt_data)

# 生成数据集
num_samples = 10000
src_data, tgt_data = create_dataset(num_samples)

def positional_encoding(seq_len, dim_model, device):
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / (1e4 ** (dim // dim_model))
    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))

def attention(query, key, value, mask=None):
    dim_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, value)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_model):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_k = dim_model // num_heads

        self.w_q = nn.Linear(dim_model, dim_model)
        self.w_k = nn.Linear(dim_model, dim_model)
        self.w_v = nn.Linear(dim_model, dim_model)
        self.w_o = nn.Linear(dim_model, dim_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.w_q(query).view(batch_size, -1, self.num_heads, self.dim_k).transpose(1, 2)
        key = self.w_k(key).view(batch_size, -1, self.num_heads, self.dim_k).transpose(1, 2)
        value = self.w_v(value).view(batch_size, -1, self.num_heads, self.dim_k).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        scores = attention(query, key, value, mask)

        scores = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_k)
        return self.w_o(scores)

class FeedForward(nn.Module):
    def __init__(self, dim_model, dim_ff):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(dim_model, dim_ff)
        self.w_2 = nn.Linear(dim_ff, dim_model)

    def forward(self, x):
        return self.w_2(F.relu(self.w_1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, dim_model, num_heads, dim_ff):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(num_heads, dim_model)
        self.feed_forward = FeedForward(dim_model, dim_ff)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)

    def forward(self, x, mask):
        attended = self.attention(x, x, x, mask)
        x = self.norm1(attended + x)

        fed = self.feed_forward(x)
        x = self.norm2(fed + x)

        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, num_layers, dim_model, num_heads, dim_ff, device):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim_model)
        self.pos_encoding = positional_encoding(vocab_size, dim_model, device)
        self.layers = nn.ModuleList([EncoderLayer(dim_model, num_heads, dim_ff) for _ in range(num_layers)])

    def forward(self, x, mask):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        for layer in self.layers:
            x = layer(x, mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, dim_model, num_heads, dim_ff):
        super(DecoderLayer, self).__init__()
        self.attention1 = MultiHeadAttention(num_heads, dim_model)
        self.attention2 = MultiHeadAttention(num_heads, dim_model)
        self.feed_forward = FeedForward(dim_model, dim_ff)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attended1 = self.attention1(x, x, x, tgt_mask)
        x = self.norm1(attended1 + x)

        attended2 = self.attention2(x, enc_output, enc_output, src_mask)
        x = self.norm2(attended2 + x)

        fed = self.feed_forward(x)
        x = self.norm3(fed + x)

        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, num_layers, dim_model, num_heads, dim_ff, device):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim_model)
        self.pos_encoding = positional_encoding(vocab_size, dim_model, device)
        self.layers = nn.ModuleList([DecoderLayer(dim_model, num_heads, dim_ff) for _ in range(num_layers)])
        self.fc = nn.Linear(dim_model, vocab_size)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.fc(x)

# 超参数
vocab_size = len(chars)
dim_model = 128
num_layers = 3
num_heads = 8
dim_ff = 512
batch_size = 64
num_epochs = 10
learning_rate = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化编码器和解码器
encoder = Encoder(vocab_size, num_layers, dim_model, num_heads, dim_ff, device)
decoder = Decoder(vocab_size, num_layers, dim_model, num_heads, dim_ff, device)

encoder.to(device)
decoder.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

# 训练循环
for epoch in range(num_epochs):
    for i in range(0, num_samples, batch_size):
        batch_src = src_data[i:i+batch_size].to(device)
        batch_tgt = tgt_data[i:i+batch_size].to(device)

        src_mask = torch.ones(batch_src.size(0), 1, batch_src.size(1)).to(device)
        tgt_mask = torch.tril(torch.ones(batch_tgt.size(0), batch_tgt.size(1), batch_tgt.size(1))).to(device)

        enc_output = encoder(batch_src, src_mask)
        output = decoder(batch_tgt[:, :-1], enc_output, src_mask, tgt_mask[:, :-1, :-1])

        loss = criterion(output.reshape(-1, vocab_size), batch_tgt[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# 测试模型
src_seq = "hello"
src_tensor = torch.tensor([[char2index(c) for c in src_seq]]).to(device)
src_mask = torch.ones(1, 1, len(src_seq)).to(device)

enc_output = encoder(src_tensor, src_mask)
tgt_tensor = torch.zeros(1, len(src_seq), dtype=torch.long).to(device)

for i in range(len(src_seq)):
    tgt_mask = torch.tril(torch.ones(1, i+1, i+1)).to(device)
    output = decoder(tgt_tensor[:, :i+1], enc_output, src_mask, tgt_mask)
    pred_token = output.argmax(dim=-1)[:, -1].item()
    tgt_tensor[:, i] = pred_token

predicted_seq = ''.join([index2char(idx.item()) for idx in tgt_tensor[0]])
print(f"Source: {src_seq}")
print(f"Predicted: {predicted_seq}")