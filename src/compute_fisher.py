"""Compute diagonal empirical Fisher information for a CIFAR-10 ResNet-20 model."""
import argparse
import os
import pickle

import augmax
import jax
import jax.nn
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from flax.serialization import from_bytes
from jax import grad, jit, random
from tqdm import tqdm

from resnet20 import BLOCKS_PER_GROUP, ResNet
from utils import flatten_params

# See https://github.com/google/jax/issues/9454.
tf.config.set_visible_devices([], "GPU")

NUM_CLASSES = 10


def compute_diagonal_fisher(model, params, dataset, num_samples=1000, seed=42):
    """Compute empirical diagonal Fisher: F_k = (1/N) sum (dL/dtheta_k)^2."""
    rng = np.random.default_rng(seed)
    num_total = dataset["images_u8"].shape[0]
    indices = rng.choice(num_total, size=num_samples, replace=False)

    # Use the EXACT same normalization as training pipeline
    normalize_transform = augmax.Chain(augmax.ByteToFloat(), augmax.Normalize())

    def single_loss(params, image_u8, label):
        image_f32 = normalize_transform(None, image_u8)
        logits = model.apply({"params": params}, image_f32[None])
        y_onehot = jax.nn.one_hot(label, NUM_CLASSES)
        return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y_onehot[None]))

    grad_fn = jit(grad(single_loss))

    # Accumulate squared gradients
    fisher = jax.tree_map(jnp.zeros_like, params)
    for idx in tqdm(indices, desc="Computing Fisher"):
        image = dataset["images_u8"][idx]
        label = dataset["labels"][idx]
        g = grad_fn(params, image, label)
        fisher = jax.tree_map(lambda f, g: f + g**2, fisher, g)

    # Average
    fisher = jax.tree_map(lambda f: f / num_samples, fisher)
    return fisher


def normalize_fisher_per_matrix(fisher_flat):
    """Per-matrix normalization: F_k -> F_k / mean(F_k) per weight matrix."""
    normalized = {}
    for k, v in fisher_flat.items():
        mean_val = jnp.mean(v)
        normalized[k] = v / (mean_val + 1e-12)
    return normalized


if __name__ == "__main__":
    from datasets import load_cifar10

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width-multiplier", type=int, default=1)
    args = parser.parse_args()

    model = ResNet(blocks_per_group=BLOCKS_PER_GROUP["resnet20"],
                   num_classes=NUM_CLASSES,
                   width_multiplier=args.width_multiplier)

    with open(args.checkpoint, "rb") as fh:
        params = from_bytes(
            model.init(random.PRNGKey(0), jnp.zeros((1, 32, 32, 3)))["params"],
            fh.read())

    train_ds, _ = load_cifar10()

    fisher = compute_diagonal_fisher(model, params, train_ds,
                                     num_samples=args.num_samples, seed=args.seed)
    fisher_flat = flatten_params(fisher)
    fisher_normalized = normalize_fisher_per_matrix(fisher_flat)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({"raw": fisher_flat, "normalized": fisher_normalized}, f)

    print(f"Fisher saved to {args.output} ({len(fisher_flat)} parameter groups)")
