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
        # 随机生成5个大写字母
        src = ''.join(random.choices(chars, k=5))
        # 将大写字母转换为小写字母
        tgt = src.lower()
        src_data.append([char2index(c) for c in src])
        tgt_data.append([char2index(c) for c in tgt])
    return torch.tensor(src_data), torch.tensor(tgt_data)


# 生成数据集
num_samples = 10000
src_data, tgt_data = create_dataset(num_samples)


def positional_encoding(seq_len, dim_model, device):
    """
    位置编码函数,用于将位置信息编码到输入的词向量中。

    参数:
    seq_len: 序列的长度
    dim_model: 词向量的维度
    device: 设备(CPU或GPU)

    返回:
    pos_encoding: 位置编码张量,形状为(1, seq_len, dim_model)
    """
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / (1e4 ** (dim // dim_model))
    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))


def attention(query, key, value, mask=None):
    """
    注意力机制函数,用于计算注意力权重并应用到值上。

    参数:
    query: 查询张量,形状为(batch_size, num_heads, seq_len, dim_k)
    key: 键张量,形状为(batch_size, num_heads, seq_len, dim_k)
    value: 值张量,形状为(batch_size, num_heads, seq_len, dim_k)
    mask: 掩码张量,形状为(batch_size, num_heads, seq_len, seq_len)

    返回:
    attended: 注意力输出张量,形状为(batch_size, num_heads, seq_len, dim_k)
    """
    dim_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, value)


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制模块。

    参数:
    num_heads: 注意力头的数量
    dim_model: 词向量的维度
    """

    def __init__(self, num_heads, dim_model):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_k = dim_model // num_heads

        self.w_q = nn.Linear(dim_model, dim_model)
        self.w_k = nn.Linear(dim_model, dim_model)
        self.w_v = nn.Linear(dim_model, dim_model)
        self.w_o = nn.Linear(dim_model, dim_model)

    def forward(self, query, key, value, mask=None):
        """
        前向传播函数。

        参数:
        query: 查询张量,形状为(batch_size, seq_len, dim_model)
        key: 键张量,形状为(batch_size, seq_len, dim_model)
        value: 值张量,形状为(batch_size, seq_len, dim_model)
        mask: 掩码张量,形状为(batch_size, seq_len, seq_len)

        返回:
        output: 注意力输出张量,形状为(batch_size, seq_len, dim_model)
        """
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
    """
    前馈神经网络模块。

    参数:
    dim_model: 词向量的维度
    dim_ff: 前馈神经网络的隐藏层维度
    """

    def __init__(self, dim_model, dim_ff):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(dim_model, dim_ff)
        self.w_2 = nn.Linear(dim_ff, dim_model)

    def forward(self, x):
        """
        前向传播函数。

        参数:
        x: 输入张量,形状为(batch_size, seq_len, dim_model)

        返回:
        output: 输出张量,形状为(batch_size, seq_len, dim_model)
        """
        return self.w_2(F.relu(self.w_1(x)))


class EncoderLayer(nn.Module):
    """
    编码器层模块。

    参数:
    dim_model: 词向量的维度
    num_heads: 注意力头的数量
    dim_ff: 前馈神经网络的隐藏层维度
    """

    def __init__(self, dim_model, num_heads, dim_ff):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(num_heads, dim_model)
        self.feed_forward = FeedForward(dim_model, dim_ff)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)

    def forward(self, x, mask):
        """
        前向传播函数。

        参数:
        x: 输入张量,形状为(batch_size, seq_len, dim_model)
        mask: 掩码张量,形状为(batch_size, seq_len, seq_len)

        返回:
        output: 输出张量,形状为(batch_size, seq_len, dim_model)
        """
        attended = self.attention(x, x, x, mask)
        x = self.norm1(attended + x)

        fed = self.feed_forward(x)
        x = self.norm2(fed + x)

        return x


class Encoder(nn.Module):
    """
    编码器模块。

    参数:
    vocab_size: 词汇表大小
    num_layers: 编码器层的数量
    dim_model: 词向量的维度
    num_heads: 注意力头的数量
    dim_ff: 前馈神经网络的隐藏层维度
    """

    def __init__(self, vocab_size, num_layers, dim_model, num_heads, dim_ff, device):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim_model)
        self.pos_encoding = positional_encoding(vocab_size, dim_model, device)
        self.layers = nn.ModuleList([EncoderLayer(dim_model, num_heads, dim_ff) for _ in range(num_layers)])

    def forward(self, x, mask):
        """
        前向传播函数。

        参数:
        x: 输入张量,形状为(batch_size, seq_len)
        mask: 掩码张量,形状为(batch_size, seq_len, seq_len)

        返回:
        output: 编码器输出张量,形状为(batch_size, seq_len, dim_model)
        """
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecoderLayer(nn.Module):
    """
    解码器层模块。

    参数:
    dim_model: 词向量的维度
    num_heads: 注意力头的数量
    dim_ff: 前馈神经网络的隐藏层维度
    """

    def __init__(self, dim_model, num_heads, dim_ff):
        super(DecoderLayer, self).__init__()
        self.attention1 = MultiHeadAttention(num_heads, dim_model)
        self.attention2 = MultiHeadAttention(num_heads, dim_model)
        self.feed_forward = FeedForward(dim_model, dim_ff)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        """
        前向传播函数。

        参数:
        x: 输入张量,形状为(batch_size, seq_len, dim_model)
        enc_output: 编码器输出张量,形状为(batch_size, seq_len, dim_model)
        src_mask: 源序列掩码张量,形状为(batch_size, seq_len, seq_len)
        tgt_mask: 目标序列掩码张量,形状为(batch_size, seq_len, seq_len)

        返回:
        output: 输出张量,形状为(batch_size, seq_len, dim_model)
        """
        attended1 = self.attention1(x, x, x, tgt_mask)
        x = self.norm1(attended1 + x)

        attended2 = self.attention2(x, enc_output, enc_output, src_mask)
        x = self.norm2(attended2 + x)

        fed = self.feed_forward(x)
        x = self.norm3(fed + x)

        return x


class Decoder(nn.Module):
    """
    解码器模块。

    参数:
    vocab_size: 词汇表大小
    num_layers: 解码器层的数量
    dim_model: 词向量的维度
    num_heads: 注意力头的数量
    dim_ff: 前馈神经网络的隐藏层维度
    """

    def __init__(self, vocab_size, num_layers, dim_model, num_heads, dim_ff, device):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim_model)
        self.pos_encoding = positional_encoding(vocab_size, dim_model, device)
        self.layers = nn.ModuleList([DecoderLayer(dim_model, num_heads, dim_ff) for _ in range(num_layers)])
        self.fc = nn.Linear(dim_model, vocab_size)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        """
        前向传播函数。

        参数:
        x: 输入张量,形状为(batch_size, seq_len)
        enc_output: 编码器输出张量,形状为(batch_size, seq_len, dim_model)
        src_mask: 源序列掩码张量,形状为(batch_size, seq_len, seq_len)
        tgt_mask: 目标序列掩码张量,形状为(batch_size, seq_len, seq_len)

        返回:
        output: 解码器输出张量,形状为(batch_size, seq_len, vocab_size)
        """
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.fc(x)


# 超参数
vocab_size = len(chars)
dim_model = 128
num_layers = 5
num_heads = 8
dim_ff = 512
batch_size = 64
num_epochs = 10
learning_rate = 0.001
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
# 训练循环
# 训练循环
for epoch in range(num_epochs):
    for i in range(0, num_samples, batch_size):
        batch_src = src_data[i:i+batch_size].to(device)
        batch_tgt = tgt_data[i:i+batch_size].to(device)

        # 创建掩码
        src_mask = torch.ones(batch_src.size(0), 1, batch_src.size(1)).to(device)
        tgt_mask = torch.tril(torch.ones(batch_tgt.size(0), batch_tgt.size(1), batch_tgt.size(1))).to(device)

        # 前向传播
        enc_output = encoder(batch_src, src_mask)
        output = decoder(batch_tgt[:, :-1], enc_output, src_mask, tgt_mask[:, :-1, :-1])

        # 计算损失
        loss = criterion(output.reshape(-1, vocab_size), batch_tgt[:, 1:].reshape(-1))

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
