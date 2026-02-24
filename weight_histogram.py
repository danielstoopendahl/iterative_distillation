import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def load_state_dict(checkpoint_path: Path):
	checkpoint = torch.load(checkpoint_path, map_location="cpu")

	if isinstance(checkpoint, dict):
		if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
			return checkpoint["state_dict"]
		if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
			return checkpoint

	raise ValueError(
		f"Unsupported checkpoint format in {checkpoint_path}. "
		"Expected a state_dict or a dict containing 'state_dict'."
	)


def collect_parameter_magnitudes(state_dict: dict):
	magnitudes = []
	for tensor in state_dict.values():
		if torch.is_tensor(tensor) and torch.is_floating_point(tensor):
			magnitudes.append(tensor.detach().abs().view(-1))

	if not magnitudes:
		raise ValueError("No floating-point parameters found in checkpoint.")

	return torch.cat(magnitudes).cpu().numpy()


def resolve_model_path(model_name: str) -> Path:
	filename = model_name if model_name.endswith(".pth") else f"{model_name}.pth"
	return Path("models") / filename


def main():
	parser = argparse.ArgumentParser(
		description="Plot histogram of parameter magnitudes from a PyTorch checkpoint"
	)
	parser.add_argument(
		"--model",
		type=str,
		default="cifar10_cnn.pth",
		help="Model filename in models/ (with or without .pth)",
	)
	parser.add_argument(
		"--bins",
		type=int,
		default=100,
		help="Number of histogram bins",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Output image path (default: histograms/weight_histogram_<model>.png)",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="Display the histogram window",
	)
	args = parser.parse_args()
	checkpoint_path = resolve_model_path(args.model)

	if not checkpoint_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

	if args.output is None:
		output_path = Path("histograms") / f"weight_histogram_{checkpoint_path.stem}.png"
	else:
		output_path = args.output

	output_path.parent.mkdir(parents=True, exist_ok=True)

	state_dict = load_state_dict(checkpoint_path)
	magnitudes = collect_parameter_magnitudes(state_dict)

	plt.figure(figsize=(10, 6))
	plt.hist(magnitudes, bins=args.bins)
	plt.title(f"Parameter Magnitudes: {checkpoint_path}")
	plt.xlabel("|parameter value|")
	plt.ylabel("Count")
	plt.tight_layout()
	plt.savefig(output_path, dpi=150)
	print(f"Saved histogram to: {output_path}")

	if args.show:
		plt.show()
	else:
		plt.close()


if __name__ == "__main__":
	main()
