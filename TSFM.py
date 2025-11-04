import torch
import torch.nn as nn
import torch.nn.functional as F
class MultiHeadAttention(nn.Module):
    def __init__(self,dim,n_head,dropout=0.1):
        super(MultiHeadAttention,self).__init__()
        self.dim = dim
        self.head_dim = dim//n_head
        self.n_head = n_head
        assert self.dim == self.head_dim*n_head
        self.linear_q = nn.Linear(dim,dim)
        self.linear_k = nn.Linear(dim,dim)
        self.linear_v = nn.Linear(dim,dim)
        self.dropout=nn.Dropout(dropout)
        self.fc_out = nn.Linear(dim,dim)

    def forward(self, x, mask = None):
        b,t,d = x.size()
        Q = self.linear_q(x)
        K = self.linear_k(x)
        V = self.linear_v(x)

        Q = Q.view(b,t,self.n_head,self.head_dim).transpose(1,2)
        K = K.view(b,t,self.n_head,self.head_dim).transpose(1,2)
        V = V.view(b,t,self.n_head,self.head_dim).transpose(1,2)

        score = torch.matmul(Q,K.transpose(2,3))/torch.sqrt(torch.tensor(self.head_dim),dtype = torch.float32)

        if mask is not None:
            mask = mask.unsqueeze(1)
            score = score.masked_fill(mask == 0, -1e9)
        score = F.softmax(score,dim=-1)
        if self.dropout is not None:
            score = self.dropout(score)

        output = torch.matmul(score,V).transpose(1,2).contiguous().view(b,t,d)
        output = self.fc_out(output)
        return output
##### 对于decoder部分，生成下三角矩阵
def generate_mask(len_seq):
    mask = torch.tril(torch.ones(len_seq, len_seq))
    # 将1转换为True，0转换为False
    # mask = mask.bool()
    return mask

embed_dim = 512
num_heads = 8
model = MultiHeadAttention(embed_dim, num_heads)
x = torch.randn(16, 10, 512)  # 批量大小为16，序列长度为10，特征维度为512
# 生成对角掩码
mask = generate_mask(10).unsqueeze(0).expand(16, 10, 10)
output = model(x, mask)
print(output.shape)  # 输出维度应为[16, 10, 512]