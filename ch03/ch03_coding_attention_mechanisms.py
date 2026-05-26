#!/usr/bin/env python
# coding: utf-8

# # Chapter 3 Coding attention mechanism

# In[1]:


import torch
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)


# In[2]:


inputs[1]


# In[3]:


query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)


# In[4]:


attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())


# In[5]:


attn_weights_2_tmp


# In[6]:


attn_scores_2[0] / attn_scores_2.sum()


# In[7]:


def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights: ", attn_weights_2_naive)
print("Sum: ", attn_weights_2_naive.sum())


# In[8]:


0.4*0.5+0.1*0.8+0.8*0.6


# In[9]:


attn_scores_2


# In[10]:


torch.exp(torch.tensor([20]))


# In[11]:


attn_scores_2


# In[12]:


attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights: ", attn_weights_2)
print("Sum: ", attn_weights_2.sum())


# In[13]:


inputs[1]


# In[14]:


print(attn_weights_2)


# In[15]:


torch.zeros(query.shape)


# In[16]:


query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    print("i = ", i)
    print("x_i = ", x_i)
    print("attn_weights_2[i] = ", attn_weights_2[i])
    print("context_vec_2_tmp = ", attn_weights_2[i]*x_i)
    context_vec_2 += attn_weights_2[i]*x_i
    print("\n")
print(context_vec_2)


# In[17]:


# copmputing all attention scores with a for loop, which is slower than matrix multiplication

attn_scores = torch.empty(6,6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)


# In[18]:


# computing all attention scores with matrix multiplication

attn_scores = inputs @ inputs.T
print(attn_scores)


# In[19]:


attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)


# In[20]:


row_2_sum = sum([0.2098, 0.2006, 0.1981, 0.1242, 0.1220, 0.1452])
print("Row 2 sum: ", row_2_sum)
print("Row 2 sum: ", attn_weights[1].sum(dim=-1))
print("All row sums: ", attn_weights.sum(dim=-1))


# In[21]:


all_context_vecs = attn_weights @ inputs
print(all_context_vecs)


# In[22]:


print("Previous 2nd context vector: ", context_vec_2)


# ## 3.4 - Implementing self-attention with trainable weights

# ### 3.4.1 Computing the attention weights step by step

# In[23]:


x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2


# In[24]:


x_2


# In[25]:


print(d_in)


# In[26]:


torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)


# In[27]:


query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)


# In[28]:


keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape: ", keys.shape)
print("values.shape: ", values.shape)


# In[29]:


keys


# In[30]:


values


# In[31]:


keys[1]


# In[32]:


keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)


# In[33]:


sum(query_2 * keys_2)


# In[34]:


query_2 @ keys_2


# In[35]:


attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)


# In[36]:


d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)


# In[37]:


context_vec_2 = attn_weights_2 @ values
print(context_vec_2)


# ### 3.4.2 Implementing a compact self-attention Python class

# In[38]:


import torch.nn as nn
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec


# In[39]:


torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))


# In[40]:


torch.manual_seed(789)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))


# In[41]:


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec


# In[42]:


torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))



# In[44]:


test = sa_v1.W_query


# In[45]:


test = sa_v2.W_query.weight.T


# In[46]:


test


# In[47]:


sa_v1.W_query = nn.Parameter(sa_v2.W_query.weight.T)
sa_v1.W_value = nn.Parameter(sa_v2.W_value.weight.T)
sa_v1.W_key = nn.Parameter(sa_v2.W_key.weight.T)


# In[48]:


print(sa_v1(inputs))
print(sa_v2(inputs))


# ## 3.5 Hiding future words with causal attention

# In[49]:


queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)


# In[50]:


print(attn_scores)


# In[51]:


context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)


# In[52]:


masked_simple = attn_weights * mask_simple
print(masked_simple)


# In[53]:


row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)


# In[54]:


mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)


# In[55]:


attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print(attn_weights)


# In[56]:


print(torch.ones(6,6))


# In[57]:


torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6,6)
print(dropout(example))


# In[58]:


torch.manual_seed(123)
print(dropout(attn_weights))


# In[59]:


batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)


# In[60]:


print(batch)


# In[61]:


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),diagonal=1)
        )
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec


# In[62]:


torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)


# In[63]:


print(context_vecs)


# In[64]:


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(
                d_in, d_out, context_length, dropout, qkv_bias
            )
            for _ in range(num_heads)]
    )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


# In[65]:


torch.manual_seed(123)
context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)
context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)


# In[66]:


torch.manual_seed(123)
context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 1
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)
context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)


# #### 3.6.2 Implementing multi-head attention with weight splits

# In[81]:


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
        "d_out must be divisible by num_heads"

        self.d_out = d_out 
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(
            b, num_tokens, self.num_heads, self.head_dim
        )

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        context_vec = self.out_proj(context_vec)
        return context_vec


# In[82]:


# (b, num_heads, num_tokens, head_dim) = (1, 2, 3, 4)
a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340]],

                   [[0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786]]]])


# In[83]:


a


# In[84]:


a.shape


# In[85]:


a.transpose(2, 3)


# In[86]:


print(a @ a.transpose(2, 3))


# In[87]:


first_head = a[0, 0, :, :]
first_res = first_head @ first_head.T
print("First head:\n", first_res)

second_head = a[0, 1, :, :]
second_res = second_head @ second_head.T
print("\nSecond head:\n", second_res)


# In[88]:


torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)


# In[91]:


torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 768
context_length = 1024
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=12)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)


# In[ ]:




