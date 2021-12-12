import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from CONTROLLER.models import EncoderLayer

class GNN(nn.Module):
	"""
	Graph Attention
	"""
	def __init__(self, input_dim, gat_output_dim=128, nheads=8, node=36, dropout=0, act_dim=10):
		super(GNN, self).__init__()
		self.emb = EncoderLayer(input_dim, output_dim=gat_output_dim, nheads=8, node=node, dropout=0)
		
		self.down = nn.Linear(gat_output_dim, 1)
		self.mu = nn.Linear(gat_output_dim, act_dim)

	def forward(self, x, adj):
		x_orig = x
		p1, p2 = x[:,:,:3], x[:,:,4:]
		x = torch.cat([p1, p2], dim=-1)
		t = x[:,:,3]
		x = self.emb(x, adj) # output: [1, 177, 128]
		x = self.down(x).squeeze(-1) # [1, 177]
		x = torch.cat([x, t], dim=-1) # [1, 354]

		x = F.leaky_relu(x)

		mu = self.mu(x) # [1, 10]
		return mu


class FFN(nn.Module):
	"""
		A standard Feed Forward Neural Network.
	"""
	def __init__(self, observation_size, action_size, model_dim):
		"""
			Initialize the network and set up the layers.
			Parameters:
				observation_size - output dimensions as an int
				action_size - input dimensions as an int
			Return:
				None
		"""
		super(FFN, self).__init__()

		self.in_dim = observation_size
		self.out_dim = action_size
		self.model_dim = model_dim

		self.layer1 = nn.Linear(self.in_dim, self.model_dim)
		self.layer2 = nn.Linear(self.model_dim, self.model_dim)
		self.layer3 = nn.Linear(self.model_dim, self.out_dim)

	def forward(self, state):
		"""
			Runs a forward pass on the neural network.
			Parameters:
				state - converted observation to pass as input
			Return:
				out - the output of our forward pass
		"""
		# Convert observation to tensor if it's a numpy array

		x = F.relu(self.layer1(state))
		x = F.relu(self.layer2(x))
		out = self.layer3(x)

		return out

	