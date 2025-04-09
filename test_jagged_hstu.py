import torch
from tqdm import tqdm
import random
from fused_hstu_v3.fused_hstu_op_v3 import FusedHSTUOpv3

def get_input(sum_N, head, d, B, n):
    q = torch.randn(sum_N, head*d, requires_grad=True, device="cuda")
    k = torch.randn(sum_N, head*d, requires_grad=True, device="cuda")
    v = torch.randn(sum_N, head*d, requires_grad=True, device="cuda")
    rab = torch.randn(B, 1, n, n, requires_grad=True, device="cuda")

    q1 = q.clone().detach().requires_grad_(True)
    k1 = k.clone().detach().requires_grad_(True)
    v1 = v.clone().detach().requires_grad_(True)
    rab1 = rab.clone().detach().requires_grad_(True)
    
    # 生成一个下三角矩阵
    attn_mask = torch.tril(torch.ones((n, n), device='cuda:0'))
    # 调整形状为 (1, 1, n, n)
    attn_mask = attn_mask.view(1, 1, n, n) 
    return q, k, v, rab,  q1, k1, v1, rab1, attn_mask

seq_len = [120,128, 256, 512, 1024]
max_seq = 200
min_seq = 100
n = 0
B  = 128
x_offsets = [0]
for i in range(1, B+1):
    # rand_seq_len = random.choice(seq_len)
    rand_seq_len = random.randint(min_seq, max_seq)
    n = max(n, rand_seq_len)
    x_offsets.append(x_offsets[-1] + rand_seq_len) # 生成一个长度为B的序列，每个元素为0-1024之间的随机数
x_offsets = torch.tensor(x_offsets, device="cuda") # 转换为tensor

n += 11  #符合原本hstu的流程
head, d = 2 , 25
sum_N = int(x_offsets[-1])

print('config: sum_N: {}, head: {}, d: {}, B: {}, n: {}'.format(sum_N, head, d, B, n))
print('input q k v & output shape: ({}, {})'.format(sum_N, head*d))
print('input rab shape: ({}, {}, {}, {})'.format(B, 1, n, n))
print('input attn_mask shape: ({}, {}, {}, {})'.format(1, 1, n, n))

print('===========================================================')

test_num = 10
for _ in tqdm(range(test_num)):
    q, k, v, rab, q1, k1, v1, rab1, attn_mask = get_input(sum_N, head, d, B, n)

    fused_attn = FusedHSTUOpv3.apply(q1, k1, v1, rab1, attn_mask, head, d, n, x_offsets)
    y_true = torch.randn_like(fused_attn)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(fused_attn, y_true)
    loss.backward()


