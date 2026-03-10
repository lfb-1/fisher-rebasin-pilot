"""Run weight matching + interpolation evaluation for Fisher Re-Basin pilot."""
import argparse
import csv
import os
import pickle

import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from einops import rearrange
from flax import linen as nn
from flax.serialization import from_bytes
from jax import jit, random, vmap
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from cifar10_resnet20_train import make_stuff, NUM_CLASSES
from datasets import load_cifar10
from online_stats import OnlineCovariance, OnlineMean
from resnet20 import BLOCKS_PER_GROUP, ResNet
from utils import flatten_params, unflatten_params, lerp
from weight_matching import (apply_permutation, resnet20_permutation_spec,
                              weight_matching, weight_matching_custom_cost)

# See https://github.com/google/jax/issues/9454.
tf.config.set_visible_devices([], "GPU")

EPS = 1e-8


def load_model(model, filepath):
    with open(filepath, "rb") as fh:
        return from_bytes(
            model.init(random.PRNGKey(0), jnp.zeros((1, 32, 32, 3)))["params"],
            fh.read())


def load_fisher(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def prescale_params(params_flat, weights_flat):
    """Pre-scale parameters by sqrt(weights) for inner-product methods."""
    scaled = {}
    for k in params_flat:
        if k in weights_flat:
            scaled[k] = params_flat[k] * jnp.sqrt(weights_flat[k] + EPS)
        else:
            scaled[k] = params_flat[k]
    return scaled


def compute_magnitude_weights(params_a_flat, params_b_flat):
    """M_k = (|W^A_k|^2 + |W^B_k|^2) / 2, per-matrix normalized."""
    mag = {}
    for k in params_a_flat:
        m = (params_a_flat[k]**2 + params_b_flat[k]**2) / 2.0
        mean_m = jnp.mean(m)
        mag[k] = m / (mean_m + 1e-12)
    return mag


def compute_activation_init_perm(model, params_a, params_b, train_ds, ps, num_examples=500):
    """Compute init_perm for outer permutation groups using activation correlations.

    Collects activations at block group boundaries (after norm1/relu for P_bg0,
    after each blockgroup for P_bg1, P_bg2) and uses correlation-based LAP
    to initialize the 3 outer permutation groups.
    """
    stuff = make_stuff(model)
    normalize_transform = stuff["normalize_transform"]

    batch_size = 500
    num_batches = num_examples // batch_size

    # Permute training data for subset selection
    train_perm = np.random.default_rng(123).permutation(train_ds["images_u8"].shape[0])
    subset_indices = train_perm[:num_examples]

    # We need activations at block group boundaries.
    # For ResNet-20: conv1 -> norm1 -> relu -> bg0 -> bg1 -> bg2 -> pool -> dense
    # P_bg0: output channels of conv1/norm1, shared by all blocks in bg0
    # P_bg1: output channels of bg0's last block / bg1's shortcut
    # P_bg2: output channels of bg1's last block / bg2's shortcut
    #
    # We use capture_intermediates to get LayerNorm outputs at the right places.
    # After norm1 (before blockgroups): gives us P_bg0 activations
    # After blockgroups_0: gives us P_bg1 transition activations
    # After blockgroups_1: gives us P_bg2 transition activations

    def get_block_boundary_activations(params, images_u8):
        """Get activations at block group boundaries using manual forward pass."""
        images_f32 = vmap(normalize_transform)(None, images_u8)
        vars = {"params": params}
        bound = model.bind(vars)

        # conv1 -> norm1 -> relu -> P_bg0
        h = bound.conv1(images_f32)
        h = bound.norm1(h)
        act_bg0 = nn.relu(h)  # shape: (batch, H, W, 16*wm)

        # blockgroup 0 output -> P_bg1 transition
        act_after_bg0 = bound.blockgroups[0](act_bg0)

        # blockgroup 1 output -> P_bg2 transition
        act_after_bg1 = bound.blockgroups[1](act_after_bg0)

        return act_bg0, act_after_bg0, act_after_bg1

    get_acts = jit(get_block_boundary_activations)

    # Determine activation sizes from model config
    wm = model.width_multiplier
    act_sizes = {
        "P_bg0": 16 * wm,
        "P_bg1": 32 * wm,
        "P_bg2": 64 * wm,
    }

    # Pass 1: compute means
    a_means = {k: OnlineMean.init(v) for k, v in act_sizes.items()}
    b_means = {k: OnlineMean.init(v) for k, v in act_sizes.items()}

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        batch_indices = subset_indices[start:end]
        images_u8 = train_ds["images_u8"][batch_indices]

        a_bg0, a_bg1_out, a_bg2_out = get_acts(params_a, images_u8)
        b_bg0, b_bg1_out, b_bg2_out = get_acts(params_b, images_u8)

        # Flatten spatial dims: (batch, H, W, C) -> (batch*H*W, C)
        a_acts = {
            "P_bg0": rearrange(a_bg0, "b h w c -> (b h w) c"),
            "P_bg1": rearrange(a_bg1_out, "b h w c -> (b h w) c"),
            "P_bg2": rearrange(a_bg2_out, "b h w c -> (b h w) c"),
        }
        b_acts = {
            "P_bg0": rearrange(b_bg0, "b h w c -> (b h w) c"),
            "P_bg1": rearrange(b_bg1_out, "b h w c -> (b h w) c"),
            "P_bg2": rearrange(b_bg2_out, "b h w c -> (b h w) c"),
        }

        for k in act_sizes:
            a_means[k] = a_means[k].update(a_acts[k])
            b_means[k] = b_means[k].update(b_acts[k])

    # Pass 2: compute covariance/correlation
    cov_stats = {
        k: OnlineCovariance.init(a_means[k].mean(), b_means[k].mean())
        for k in act_sizes
    }

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        batch_indices = subset_indices[start:end]
        images_u8 = train_ds["images_u8"][batch_indices]

        a_bg0, a_bg1_out, a_bg2_out = get_acts(params_a, images_u8)
        b_bg0, b_bg1_out, b_bg2_out = get_acts(params_b, images_u8)

        a_acts = {
            "P_bg0": rearrange(a_bg0, "b h w c -> (b h w) c"),
            "P_bg1": rearrange(a_bg1_out, "b h w c -> (b h w) c"),
            "P_bg2": rearrange(a_bg2_out, "b h w c -> (b h w) c"),
        }
        b_acts = {
            "P_bg0": rearrange(b_bg0, "b h w c -> (b h w) c"),
            "P_bg1": rearrange(b_bg1_out, "b h w c -> (b h w) c"),
            "P_bg2": rearrange(b_bg2_out, "b h w c -> (b h w) c"),
        }

        for k in act_sizes:
            cov_stats[k] = cov_stats[k].update(a_acts[k], b_acts[k])

    # Solve LAP on correlation matrix for each outer group
    pa_flat = flatten_params(params_a)
    perm_sizes = {p: pa_flat[axes[0][0]].shape[axes[0][1]]
                  for p, axes in ps.perm_to_axes.items()}
    init_perm = {p: jnp.arange(n) for p, n in perm_sizes.items()}

    for k in ["P_bg0", "P_bg1", "P_bg2"]:
        corr = cov_stats[k].pearson_correlation()
        ri, ci = linear_sum_assignment(corr, maximize=True)
        assert (ri == jnp.arange(len(ri))).all()
        init_perm[k] = jnp.array(ci)

    return init_perm


def run_method(method, rng, ps, params_a_flat, params_b_flat,
               fisher_a=None, fisher_b=None, model=None,
               raw_params_a=None, raw_params_b=None, train_ds=None):
    """Run one matching method, return permutation."""
    if method == "euclidean":
        return weight_matching(rng, ps, params_a_flat, params_b_flat, silent=True)

    elif method == "magnitude":
        mag = compute_magnitude_weights(params_a_flat, params_b_flat)
        scaled_a = prescale_params(params_a_flat, mag)
        scaled_b = prescale_params(params_b_flat, mag)
        return weight_matching(rng, ps, scaled_a, scaled_b, silent=True)

    elif method == "fisher_inner":
        scaled_a = prescale_params(params_a_flat, fisher_a["normalized"])
        scaled_b = prescale_params(params_b_flat, fisher_b["normalized"])
        return weight_matching(rng, ps, scaled_a, scaled_b, silent=True)

    elif method == "fisher_l2_sym":
        return weight_matching_custom_cost(
            rng, ps, params_a_flat, params_b_flat,
            fisher_a["normalized"], fisher_b["normalized"], silent=True)

    elif method == "activation_hybrid":
        init_perm = compute_activation_init_perm(
            model, raw_params_a, raw_params_b, train_ds, ps)
        return weight_matching(rng, ps, params_a_flat, params_b_flat,
                               init_perm=init_perm, silent=True)

    else:
        raise ValueError(f"Unknown method: {method}")


def evaluate_interpolation(stuff, model_a_params, model_b_permuted_params,
                           train_ds, test_ds, num_points=25):
    """Evaluate loss/accuracy along linear interpolation."""
    lambdas = jnp.linspace(0, 1, num=num_points)
    results = []
    for lam in tqdm(lambdas, desc="  Interpolation"):
        interp_params = lerp(lam, model_a_params, model_b_permuted_params)
        train_loss, train_acc = stuff["dataset_loss_and_accuracy"](interp_params, train_ds, 1000)
        test_loss, test_acc = stuff["dataset_loss_and_accuracy"](interp_params, test_ds, 1000)
        results.append({
            "lambda": float(lam),
            "train_loss": float(train_loss),
            "test_loss": float(test_loss),
            "train_acc": float(train_acc),
            "test_acc": float(test_acc),
        })
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-id", type=int, required=True)
    parser.add_argument("--model-a", type=str, required=True)
    parser.add_argument("--model-b", type=str, required=True)
    parser.add_argument("--fisher-a", type=str, default=None)
    parser.add_argument("--fisher-b", type=str, default=None)
    parser.add_argument("--methods", nargs="+",
                        default=["euclidean", "magnitude", "fisher_inner",
                                 "fisher_l2_sym", "activation_hybrid"])
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width-multiplier", type=int, default=1)
    args = parser.parse_args()

    # Setup
    model = ResNet(blocks_per_group=BLOCKS_PER_GROUP["resnet20"],
                   num_classes=NUM_CLASSES, width_multiplier=args.width_multiplier)
    stuff = make_stuff(model)
    ps = resnet20_permutation_spec()
    train_ds, test_ds = load_cifar10()
    rng = random.PRNGKey(args.seed)

    # Load models
    model_a = load_model(model, args.model_a)
    model_b = load_model(model, args.model_b)
    params_a_flat = flatten_params(model_a)
    params_b_flat = flatten_params(model_b)

    # Load Fisher (if needed)
    fisher_a = load_fisher(args.fisher_a) if args.fisher_a else None
    fisher_b = load_fisher(args.fisher_b) if args.fisher_b else None

    # Output files
    os.makedirs(args.output_dir, exist_ok=True)
    raw_csv = os.path.join(args.output_dir, "interpolation_raw.csv")
    summary_csv = os.path.join(args.output_dir, "interpolation_summary.csv")

    for method in args.methods:
        print(f"Pair {args.pair_id}, Method: {method}")

        # Skip Fisher methods if no Fisher loaded
        if "fisher" in method and fisher_a is None:
            print(f"  Skipping {method}: no Fisher data")
            continue

        # Run matching
        perm = run_method(method, rng, ps, params_a_flat, params_b_flat,
                          fisher_a=fisher_a, fisher_b=fisher_b,
                          model=model, raw_params_a=model_a, raw_params_b=model_b,
                          train_ds=train_ds)

        # Apply permutation to ORIGINAL (unscaled) params
        params_b_perm = apply_permutation(ps, perm, params_b_flat)
        model_b_perm = unflatten_params(params_b_perm)

        # Evaluate interpolation
        results = evaluate_interpolation(stuff, model_a, model_b_perm, train_ds, test_ds)

        # Save raw results
        write_header = not os.path.exists(raw_csv)
        with open(raw_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "pair_id", "method", "lambda", "train_loss", "test_loss",
                "train_acc", "test_acc"])
            if write_header:
                writer.writeheader()
            for r in results:
                writer.writerow({"pair_id": args.pair_id, "method": method, **r})

        # Compute and save summary
        endpoint_avg_train = (results[0]["train_loss"] + results[-1]["train_loss"]) / 2
        endpoint_avg_test = (results[0]["test_loss"] + results[-1]["test_loss"]) / 2
        barrier_train = max(r["train_loss"] for r in results) - endpoint_avg_train
        barrier_test = max(r["test_loss"] for r in results) - endpoint_avg_test
        acc_05_train = results[len(results) // 2]["train_acc"]
        acc_05_test = results[len(results) // 2]["test_acc"]

        write_header = not os.path.exists(summary_csv)
        with open(summary_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "pair_id", "method", "barrier_train", "barrier_test",
                "acc_at_05_train", "acc_at_05_test",
                "endpoint_avg_loss_train", "endpoint_avg_loss_test"])
            if write_header:
                writer.writeheader()
            writer.writerow({
                "pair_id": args.pair_id, "method": method,
                "barrier_train": barrier_train, "barrier_test": barrier_test,
                "acc_at_05_train": acc_05_train, "acc_at_05_test": acc_05_test,
                "endpoint_avg_loss_train": endpoint_avg_train,
                "endpoint_avg_loss_test": endpoint_avg_test,
            })

        # Save permutation for post-hoc analysis
        perm_path = os.path.join(args.output_dir, "permutations",
                                 f"pair{args.pair_id}_{method}.pkl")
        os.makedirs(os.path.dirname(perm_path), exist_ok=True)
        with open(perm_path, "wb") as f:
            pickle.dump(perm, f)

        print(f"  barrier_train={barrier_train:.6f}, barrier_test={barrier_test:.6f}, "
              f"acc@0.5_test={acc_05_test:.4f}")
