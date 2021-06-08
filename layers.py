from torch import nn, Tensor
from typing import Tuple

class Linear_R(nn.Module):
	def __init__(self, nb_weights_i: int, nb_weights_o: int):
		super().__init__()
		self.layers = nn.Sequential(
			nn.Linear(nb_weights_i, nb_weights_o),
			nn.ReLU(True)
		)

	def forward(self, z: Tensor) -> Tensor:
		return self.layers(z)

class Conv2d(nn.Module):
	def __init__(self, nb_weights_i: int, nb_weights_o: int, kernel_size: int, stride: int = 2):
		super().__init__()
		padding = kernel_size // 2 + (kernel_size - 2 * (kernel_size // 2)) - 1
		self.layers = nn.Sequential(
			nn.Conv2d(nb_weights_i, nb_weights_o, kernel_size, stride, padding)
		)

	def forward(self, z: Tensor) -> Tensor:
		return self.layers(z)

class Conv2d_R(Conv2d):
	def __init__(self, nb_weights_i: int, nb_weights_o: int, kernel_size: int, stride: int = 2):
		super().__init__(nb_weights_i, nb_weights_o, kernel_size, stride)
		self.layers.add_module(nn.ReLU.__name__, nn.ReLU(True))

class ConvTranspose2d(nn.Module):
	def __init__(self, nb_weights_i: int, nb_weights_o: int, kernel_size: int, stride: int = 2):
		super().__init__()
		padding = kernel_size // 2 + (kernel_size - 2 * (kernel_size // 2)) - 1
		self.layers = nn.Sequential(
			nn.ConvTranspose2d(nb_weights_i, nb_weights_o, kernel_size, stride, padding, output_padding = stride - 1)
		)

	def forward(self, z: Tensor) -> Tensor:
		return self.layers(z)

class ConvTranspose2d_R(ConvTranspose2d):
	def __init__(self, nb_weights_i: int, nb_weights_o: int, kernel_size: int, stride: int = 2):
		super().__init__(nb_weights_i, nb_weights_o, kernel_size, stride)
		self.layers.add_module(nn.ReLU.__name__, nn.ReLU(True))

class Reshape(nn.Module):
	def __init__(self, out_shape: Tuple[int, int, int]):
		super().__init__()
		self.out_shape = out_shape

	def forward(self, z: Tensor) -> Tensor:
		return z.view(z.shape[0], *self.out_shape)

class Flatten(nn.Module):
	def forward(self, z: Tensor) -> Tensor:
		return z.view(z.size(0), -1)
