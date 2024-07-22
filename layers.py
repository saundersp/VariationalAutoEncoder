from torch import nn, Tensor
from typing import Callable

class Linear_R(nn.Module):
	def __init__(self, nb_weights_i: int, nb_weights_o: int) -> None:
		super().__init__()
		self.layers = nn.Sequential(
			nn.Linear(nb_weights_i, nb_weights_o),
			nn.ReLU(True)
		)

	def forward(self, z: Tensor) -> Tensor:
		return self.layers(z)

class Conv2d(nn.Module):
	def __init__(self, nb_weights_i: int, nb_weights_o: int, kernel_size: int = 3, stride: int = 2) -> None:
		super().__init__()
		padding = kernel_size // 2 + (kernel_size - 2 * (kernel_size // 2)) - 1
		self.layers = nn.Conv2d(nb_weights_i, nb_weights_o, kernel_size, stride, padding)

	def forward(self, z: Tensor) -> Tensor:
		return self.layers(z)

class Conv2d_R(Conv2d):
	def __init__(self, nb_weights_i: int, nb_weights_o: int, kernel_size: int = 3, stride: int = 2) -> None:
		super().__init__(nb_weights_i, nb_weights_o, kernel_size, stride)
		self.layers.add_module(nn.ReLU.__name__, nn.ReLU(True))

class ConvTranspose2d(nn.Module):
	def __init__(self, nb_weights_i: int, nb_weights_o: int, kernel_size: int = 3, stride: int = 2) -> None:
		super().__init__()
		padding = kernel_size // 2 + (kernel_size - 2 * (kernel_size // 2)) - 1
		self.layers = nn.ConvTranspose2d(nb_weights_i, nb_weights_o, kernel_size, stride, padding, stride - 1)

	def forward(self, z: Tensor) -> Tensor:
		return self.layers(z)

class ConvTranspose2d_R(ConvTranspose2d):
	def __init__(self, nb_weights_i: int, nb_weights_o: int, kernel_size: int = 3, stride: int = 2) -> None:
		super().__init__(nb_weights_i, nb_weights_o, kernel_size, stride)
		self.layers.add_module(nn.ReLU.__name__, nn.ReLU(True))

class LambdaModule(nn.Module):
	def __init__(self, fnc: Callable):
		super().__init__()
		self.fnc = fnc

	def forward(self, x: Tensor) -> Tensor:
		return self.fnc(x)

class Reshape(LambdaModule):
	def __init__(self, out_shape: tuple[int, int, int]) -> None:
		super().__init__(lambda z: z.view(z.size(0), *out_shape))

class Flatten(LambdaModule):
	def __init__(self) -> None:
		super().__init__(lambda z: z.view(z.size(0), -1))

