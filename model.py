from layers import Conv2d, Conv2d_R, ConvTranspose2d_R, Reshape, Linear_R, Flatten
from torch import nn, Tensor, device as Device
from typing import Callable, List, Tuple
import torch as tr

class VAE(nn.Module):
	def __init__(self, reparameterize: Callable[[Tensor, Tensor], Tensor]):
		super().__init__()
		self.reparameterize = reparameterize

	def encode(self, X: Tensor) -> Tuple[Tensor, Tensor]:
		X = self.encoder(X)
		return self.mu_layer(X), self.log_var_layer(X)

	def forward(self, X: Tensor) -> Tensor:
		mu, log_var = self.encode(X)
		Z = self.reparameterize(mu, log_var)
		return self.decode(Z), mu, log_var

class VAE_Linear(VAE):
	def __init__(self, reparameterize: Callable[[Tensor, Tensor], Tensor], weights: List[int], device: Device,
				 input_dim: int = None):
		super().__init__(reparameterize)

		# Probabilistic encoder
		self.encoder = nn.Sequential(
			Linear_R(input_dim, weights[0]),
			*[Linear_R(w_i, w_i1) for w_i, w_i1 in zip(weights, weights[1:])],
			Flatten()
		)

		self.mu_layer = nn.Linear(weights[-1], 2)
		self.log_var_layer = nn.Linear(weights[-1], 2)

		weights = weights[::-1]

		# Probabilistic decoder
		self.decode = nn.Sequential(
			Linear_R(2, weights[0]),
			*[Linear_R(w_i, w_i1) for w_i, w_i1 in zip(weights[:-1], weights[1:])],
			nn.Linear(weights[-1], input_dim),
			nn.Sigmoid()
		)

		self.to(device)

class VAE_Conv2d(VAE):
	def __init__(self, reparameterize: Callable[[Tensor, Tensor], Tensor], weights: List[int], device: Device,
				 input_shape: Tuple[int, int, int] = None, kernel_size: int = None, min_width: int = None):
		super().__init__(reparameterize)

		# Probabilistic encoder
		self.encoder = nn.Sequential(
			Conv2d_R(input_shape[-1], weights[0], kernel_size),
			*[Conv2d_R(w_i, w_i1, kernel_size) for w_i, w_i1 in zip(weights, weights[1:])]
		)

		# Getting the 1D output vector size
		out_weight = self.encoder(tr.empty((1, *input_shape[::-1])))
		out_weight = out_weight.view(out_weight.size(0), -1).shape[1]
		self.encoder.add_module(Flatten.__name__, Flatten())

		self.mu_layer = nn.Linear(out_weight, 2)
		self.log_var_layer = nn.Linear(out_weight, 2)

		weights = weights[::-1]

		# Probabilistic decoder
		self.decode = nn.Sequential(
			nn.Linear(2, min_width * min_width),
			nn.ReLU(True),
			Reshape((1, min_width, min_width)),
			ConvTranspose2d_R(1, weights[0], kernel_size, 1),
			*[ConvTranspose2d_R(w_i, w_i1, kernel_size) for w_i, w_i1 in zip(weights[:-1], weights[1:])],
			Conv2d(weights[-1], input_shape[-1], kernel_size, 1),
			nn.Sigmoid()
		)

		self.to(device)
