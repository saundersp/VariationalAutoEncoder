from torch import nn, Tensor
from typing import Callable, Final

class LayeredModule(nn.Module):
	def __init__(self, modules: list[nn.Module]) -> None:
		super().__init__()
		self.layers = nn.Sequential(*modules)

	def forward(self, z: Tensor) -> Tensor:
		return self.layers(z)

class Linear_R(LayeredModule):
	def __init__(self, nb_weights_i: int, nb_weights_o: int) -> None:
		super().__init__([
			nn.Linear(nb_weights_i, nb_weights_o),
			nn.ReLU(True)
		])

class Conv2d(LayeredModule):
	def __init__(self, nb_weights_i: int, nb_weights_o: int, kernel_size: int = 3, stride: int = 2) -> None:
		padding: Final[int] = kernel_size // 2 + (kernel_size - 2 * (kernel_size // 2)) - 1
		super().__init__([nn.Conv2d(nb_weights_i, nb_weights_o, kernel_size, stride, padding)])

class Conv2d_R(LayeredModule):
	def __init__(self, nb_weights_i: int, nb_weights_o: int, kernel_size: int = 3, stride: int = 2) -> None:
		super().__init__([
			Conv2d(nb_weights_i, nb_weights_o, kernel_size, stride),
			nn.ReLU(True)
		])

class ConvTranspose2d(LayeredModule):
	def __init__(self, nb_weights_i: int, nb_weights_o: int, kernel_size: int = 3, stride: int = 2) -> None:
		padding: Final[int] = kernel_size // 2 + (kernel_size - 2 * (kernel_size // 2)) - 1
		super().__init__([nn.ConvTranspose2d(nb_weights_i, nb_weights_o, kernel_size, stride, padding, stride - 1)])

class ConvTranspose2d_R(LayeredModule):
	def __init__(self, nb_weights_i: int, nb_weights_o: int, kernel_size: int = 3, stride: int = 2) -> None:
		super().__init__([
			ConvTranspose2d(nb_weights_i, nb_weights_o, kernel_size, stride),
			nn.ReLU(True)
		])

class LambdaModule(nn.Module):
	def __init__(self, fnc: Callable) -> None:
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
