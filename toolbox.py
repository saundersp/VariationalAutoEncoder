from torch import Tensor, nn, device as Device
from typing import Tuple, List, Dict
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import numpy as np
import torch as tr

formats = ['s', 'm', 'h', 'j', 'w', 'M', 'y']
nb = np.array([1, 60, 60, 24, 7, 4, 12])
def format_time(time: float) -> str:
	prod = nb.prod()

	s = ''
	for i in range(nb.shape[0])[::-1]:
		if time >= prod:
			res = int(time // prod)
			time %= prod
			s += f'{res}{formats[i]} '
		prod /= nb[i]

	if time > 0:
		s += f'{int(time * 1e3)}ms'

	return s.rstrip()

def showLoss(hist: Dict[str, float], figsize: Tuple[int, int] = (15, 5)) -> None:
	plt.figure(figsize = figsize)
	for label in ['Train', 'Test']:
		plt.plot(hist[f'{label.lower()}_loss'], label = label)
	plt.title('Loss over time')
	plt.ylabel('Loss')
	plt.xlabel('Epochs')
	plt.legend()
	plt.tight_layout()
	plt.show()

def showLatent(model: nn.Module, X_train: Tensor, X_test: Tensor, batch_size: int, y_train: Tensor | None = None,
			   y_test: Tensor | None = None, device: Device | None = None, s: int = 2, figsize: Tuple[int, int] = (20, 14)
			   ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
	names = ('\mu', 'log(\sigma^2)', '\mu + log(\sigma^2)')
	fig, axes = plt.subplots(2, 3, figsize = figsize)
	sizes = [X_train.shape[0], X_test.shape[0]]
	t_mu, t_log_var, t_mu_log_var = [[tr.empty((size, 2), dtype = tr.float32) for size in sizes] for _ in range(3)]

	for j, (mus, log_vars, mu_log_vars, ax1, X_t, y_t, set_name) in enumerate(zip(t_mu, t_log_var, t_mu_log_var, axes,
																				[X_train, X_test], [y_train, y_test], ['Train', 'Test'])):
		for batch_idx in range(0, sizes[j], batch_size):
			X_batch = X_t[batch_idx : batch_idx + batch_size]
			if device != None:
				X_batch = X_batch.to(device)
			if y_t is not None:
				y_batch = y_t[batch_idx : batch_idx + batch_size]
			mu, log_var = model.encode(X_batch)
			log_var, mu_log_var = log_var.mul(0.5).exp_(), mu + log_var

			for ax, elt, t_elt, sub_title in zip(ax1, [mu, log_var, mu_log_var], [mus, log_vars, mu_log_vars], names):
				elt = elt.detach().cpu()
				t_elt[batch_idx : batch_idx + batch_size] = elt
				ax.set_title(f'${sub_title}$')
				im = ax.scatter(elt[:, 0], elt[:, 1], s = s, c = y_batch if y_t is not None else '#1f77b4')

			if device != None:
				del X_batch

		if y_t is not None:
			fig.text(-0.65, 1.1, f'{set_name} set with labels', fontsize = 24, horizontalalignment = 'center',
					 verticalalignment = 'center', transform = ax.transAxes)
			for ax in ax1:
				fig.colorbar(im, ax = ax)
		else:
			fig.text(-0.65, 1.1, f'{set_name} set', fontsize = 24, horizontalalignment = 'center',
					 verticalalignment = 'center', transform = ax.transAxes)
	plt.tight_layout()
	plt.subplots_adjust(hspace = 0.2)
	plt.show()

	return t_mu, t_log_var, t_mu_log_var

def showInterpolation(model: nn.Module, output_shape: NDArray[np.uint8], device: Device, n: int,
					  interval: Tuple[float, float] = (-3.0, 3.0), figsize: Tuple[int, int] = (20, 20)) -> None:
	[w, h, c] = output_shape
	figure = np.empty([*(output_shape[:-1] * n), c])
	linespace = np.linspace(-3.0, 3.0, n)
	for i, yi in enumerate(linespace):
		for j, xi in enumerate(linespace):
			z_sample = tr.tensor([[xi, yi]], dtype = tr.float32).to(device)
			X_decoded = model.decode(z_sample).detach().cpu()
			reconstructed = X_decoded.reshape([*output_shape])
			figure[i * w : (i + 1) * w, j * h : (j + 1) * h] = reconstructed

	labels = np.round(np.arange(interval[0], interval[1] + 1e-14, (interval[1] - interval[0]) / n), 2)
	xticks, yticks = np.linspace(0, w * n, n + 1), np.linspace(0, h * n, n + 1)

	fig = plt.figure(figsize = figsize)
	plt.imshow(figure, cmap = 'gray')
	plt.xticks(xticks, labels)
	plt.yticks(yticks, labels)
	ax = plt.gca()
	ax.grid(True)
	ax2 = fig.add_subplot(111, sharex = ax, frameon = False)
	ax2.yaxis.tick_right()
	ax2.xaxis.tick_top()
	ax2.set_xticks(xticks)
	ax2.set_xticklabels(labels)
	ax2.set_yticks(yticks)
	ax2.set_yticklabels(labels)
	plt.title(f'Interpolation [{interval[0]}, {interval[1]}]²', fontsize = 16)
	plt.tight_layout()
	plt.show()
