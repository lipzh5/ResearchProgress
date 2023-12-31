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


















