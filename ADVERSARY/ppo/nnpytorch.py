import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from CONTROLLER.models import AEncoderLayer

class Actor(nn.Module):

	def __init__(self, input_dim, output_dim, model_dim, encoder):
		super().__init__()
		self.down = nn.Linear(input_dim, 1)
		self.linear = nn.Linear(354, 128)
		self.mu = nn.Linear(128, output_dim)

		self.encoder = encoder
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def forward(self, x, adj): # x is (b, 177, 128)
		"""
		need to do this before embedding
		p1, p2 = x[:,:,:3], x[:,:,4:]
		x = torch.cat([p1, p2], dim=-1)
		t = x[:,:,3] # topology
		"""
		# Get embeddings for state (without topology)
		# state_no_topo = torch.index_select(
		# 	x, dim=-1, index=torch.IntTensor([0,1,2,4,5]).to(self.device)
		# )
		# topo = torch.index_select(
		# 	x, dim=-1, index=torch.IntTensor([3]).to(self.device)
		# )
		p1, p2 = x[:,:,:3], x[:,:,4:]
		x = torch.cat([p1, p2], dim=-1)
		t = x[:,:,3]

		emb = self.encoder(x, adj)

		emb = self.down(emb).squeeze(-1) # [1, 177]
		x = torch.cat([emb, t], dim=-1) # [1, 354]
		x = self.linear(x)

		x = F.leaky_relu(x)

		mu = self.mu(x) # [1, 10]
		return mu

class Critic(nn.Module):
	def __init__(self, input_dim, output_dim, model_dim, encoder):
		super().__init__()
		self.down = nn.Linear(input_dim, 1)
		self.lin1 = nn.Linear(128, 64)
		self.lin2 = nn.Linear(64, 32)
		self.lin3 = nn.Linear(32, 6)

		self.layer1 = nn.Linear(1062, 128)
		self.layer2 = nn.Linear(128, 128)
		self.layer3 = nn.Linear(128, 1)

		self.encoder = encoder
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def forward(self, x, adj):
		# Get embeddings for state (without topology)
		# state_no_topo = torch.index_select(
		# 	x, dim=-1, index=torch.IntTensor([0,1,2,4,5]).to(self.device)
		# )
		p1, p2 = x[:,:,:3], x[:,:,4:]
		x = torch.cat([p1, p2], dim=-1)

		emb = self.encoder(
			x, adj
		) # [1, 177, 128]

		x = F.relu(self.lin1(emb)) # [1, 177, 64]
		x = F.relu(self.lin2(x)) # [1, 177, 32]
		x = F.relu(self.lin3(x)) # [1, 177, 6]

		x = x.reshape(x.shape[0], -1) # [b, 1062]
		x = F.relu(self.layer1(x)) # [b, 128]
		x = F.relu(self.layer2(x)) # [b, 128]
		x = self.layer3(x) # [b, 1]

		return x


# class GNN(nn.Module):
# 	"""
# 	Graph Attention
# 	"""
# 	def __init__(self, input_dim, gat_output_dim=128, nheads=8, node=36, dropout=0, act_dim=10):
# 		super(GNN, self).__init__()
# 		self.emb = EncoderLayer(input_dim, output_dim=gat_output_dim, nheads=8, node=node, dropout=0)
		
# 		self.down = nn.Linear(gat_output_dim, 1)
# 		self.linear = nn.Linear(354, 128)
# 		self.mu = nn.Linear(128, act_dim)

# 	def forward(self, x, adj):
# 		p1, p2 = x[:,:,:3], x[:,:,4:]
# 		x = torch.cat([p1, p2], dim=-1)
# 		t = x[:,:,3]

# 		x = self.emb(x, adj) # output: [1, 177, 128]
# 		x = self.down(x).squeeze(-1) # [1, 177]
# 		x = torch.cat([x, t], dim=-1) # [1, 354]
# 		x = self.linear(x)

# 		x = F.leaky_relu(x)

# 		mu = self.mu(x) # [1, 10]
# 		return mu


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

	