# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: Defines layer types that are commonly used for transformers.

import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class PositionalEncoding(nn.Module):
	"""
	Encodes information about the positions of the tokens in the sequence. In this case, the layer has no learnable
	parameters, since it is a simple function of sines and cosines.
	"""
	def __init__(self, embed_dim, dropout=0.1, max_len=5000):
		"""
		Construct the PositionalEncoding layer.
		:param embed_dim: the size of the embed dimension
		:param dropout: the dropout value
		:param max_len: the maximum possible length of the incoming sequence
		"""
		super().__init__()
		self.dropout = nn.Dropout(p=dropout)
		assert embed_dim % 2 == 0
		# create an array with a "batch dimension" of 1 (which will broadcast across all examples in the batch)
		pe = torch.zeros(1, max_len, embed_dim)
		pos = torch.arange(0, max_len).reshape(-1, 1)
		t = torch.pow(torch.tensor([1e-4]), torch.arange(0, embed_dim, 2)/embed_dim)
		pe[:, :, 0::2] = torch.sin(pos * t)
		pe[:, :, 1::2] = torch.cos(pos * t)
		# make sure the positional encoding will be saved with the model parameters (mostly for completeness)
		self.register_buffer('pe', pe)

	def forward(self, x):
		"""
		Element-wise add positional embeddings to the input sequence
		:param x: the sequence fed to the positional encoder model, of shape(N, S, D), where N is the batch size,
		S is the sequence length, and D is embed dim
		:return: the input sequence + positional encodings, of shape (N, S, D)
		"""
		N, S, D = x.shape
		# # create a placeholder, to be overwritten by your code below.
		# output = torch.empty((N, S, D))
		output = x + self.pe[:, :S, :]
		output = self.dropout(output)
		return output


class MultiHeadAttention(nn.Module):
	"""
	A model layer which implements a simplified version of masked attention, as introduced by "Attention Is All You Need"
	Usage:
		attn = MultiHeadAttention(embed_dim, num_heads=2)
		# self-attention
		data = torch.randn(batch_size, sequence_len, embed_dim)
		self_attn_output = attn(query=data, key=data, value=data)
		# attention using two inputs
		other_data = torch.randn(batch_size, sequence_len, embed_dim)
		attn_output = attn(query=data, key=other_data, value=other_data)
	"""
	def __init__(self, embed_dim, num_heads, dropout=0.1):
		"""
		Construct a new MultiHeadAttention layer.
		:param embed_dim: dimension of the token embedding
		:param num_heads: number of attention heads
		:param dropout: dropout probability
		"""
		super().__init__()
		assert embed_dim % num_heads == 0
		# applies the linear transformation to the incoming data
		self.key = nn.Linear(embed_dim, embed_dim)
		self.query = nn.Linear(embed_dim, embed_dim)
		self.value = nn.Linear(embed_dim, embed_dim)
		self.proj = nn.Linear(embed_dim, embed_dim)

		self.attn_drop = nn.Dropout(dropout)

		self.n_head = num_heads
		self.emd_dim = embed_dim
		self.head_dim = self.emd_dim // num_heads

	def forward(self, query, key, value, attn_mask=None):
		"""
		Calculate the masked attention output for the provided data, computing all attention heads in parallel.

		In the shape definition below, N is the batch size, S is the source sequence length, T is the target sequence length,
		and E is the embedding dimension.
		:param query: Input data to be used as the query, of shape (N, S, E)
		:param key: Input data to be used as the key, of shape (N, T, E)
		:param value: Input data to be used as the value, of shape (N, T, E)
		:param attn_mask: Array of shape (S, T) where mask[i, j] == 0 indicates token i in the source should not
		token j in the target.
		influence
		:return: Tensor of shape (N, S, E) giving the weighted combination of data in value according to the attention
		weights calculated using key and query.
		"""
		N, S, E = query.shape
		N, T, E = value.shape
		# 1. You'll want to split your shape from (N, T, E) into (N, T, H, E/H)
		# 2. The function torch.matmul allows you to do a batched matrix multiply. For example, you can do
		# (N, H, T, E/H) by (N, H, E/H, T) to yield a shape (N, H, T, T)
		# 3. For applying the attn_mask, think how the scores should be modified to prevent a value from influencing output.
		H = self.n_head
		splitted_query = self.query(query).reshape(N, S, H, E/H).transpose(1, 2)
		splitted_key = self.key(key).reshape(N, T, H, E/H).transpose(1, 2)
		splitted_value = self.value(value).reshape(N, T, H, E/H).transpose(1, 2)
		ratio = torch.matmul(splitted_query, splitted_key.transpose(2, 3))/math.sqrt(E/H) # (N, H, S, T)
		if attn_mask is not None:
			ratio = ratio.masked_fill(attn_mask == 0, -float('inf'))
		ratio_probs = F.softmax(ratio, dim=-1)
		output = self.attn_drop(ratio_probs)
		output = torch.matmul(output, splitted_value) # (N, H, S, E_)
		output = output.transpose(1, 2).reshape(N, S, -1)
		output = self.proj(output)
		return output


















