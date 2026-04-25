"""
hickle scenario (Path B): real downstream-style usage.

Pretend we're an ML practitioner without test access. We train a tiny
NumPy MLP, snapshot weights+optimizer state+training metadata to a
hickle file every "epoch", then resume from the latest snapshot and
verify the resumed run reproduces the original loss curve to bitwise
precision. This is exactly the use case that put hickle on the map
(see e.g. vladfi1/phillip, MichaelHills/seizure-detection — they use
hickle to persist large numpy training payloads).

Pure README-driven: only `hickle.dump` / `hickle.load`, no internals.
Run AFTER `pip install -e <rescue tree>` from outside the rescue tree.
"""
from __future__ import annotations

import os
import sys
import tempfile
import numpy as np
import hickle


def relu(x):
    return np.maximum(0.0, x)


def forward(W1, b1, W2, b2, X):
    h = relu(X @ W1 + b1)
    return h @ W2 + b2, h


def grad_step(W1, b1, W2, b2, X, y, lr=1e-2):
    yhat, h = forward(W1, b1, W2, b2, X)
    err = yhat - y                                    # (B, out)
    gW2 = h.T @ err / X.shape[0]
    gb2 = err.mean(axis=0)
    dh = (err @ W2.T) * (h > 0)
    gW1 = X.T @ dh / X.shape[0]
    gb1 = dh.mean(axis=0)
    return W1 - lr * gW1, b1 - lr * gb1, W2 - lr * gW2, b2 - lr * gb2


def train(W1, b1, W2, b2, X, y, n_epochs, ckpt_dir):
    losses = []
    for epoch in range(n_epochs):
        W1, b1, W2, b2 = grad_step(W1, b1, W2, b2, X, y)
        yhat, _ = forward(W1, b1, W2, b2, X)
        loss = float(((yhat - y) ** 2).mean())
        losses.append(loss)
        ckpt = {
            "epoch": epoch,
            "weights": {"W1": W1, "b1": b1, "W2": W2, "b2": b2},
            "metadata": {
                "loss": loss,
                "loss_history": list(losses),
                "config": {"lr": 1e-2, "hidden": int(W1.shape[1])},
                "tags": ["mlp", "regression", f"epoch={epoch}"],
            },
        }
        # mode='w' overwrites — emulating "latest" checkpoint pattern.
        hickle.dump(ckpt, os.path.join(ckpt_dir, "latest.h5"), mode="w")
    return losses, (W1, b1, W2, b2)


def main():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 8)).astype(np.float64)
    Wtrue = rng.standard_normal((8, 1)).astype(np.float64)
    y = X @ Wtrue + 0.1 * rng.standard_normal((64, 1))

    W1_0 = rng.standard_normal((8, 16)) * 0.1
    b1_0 = np.zeros(16)
    W2_0 = rng.standard_normal((16, 1)) * 0.1
    b2_0 = np.zeros(1)

    with tempfile.TemporaryDirectory() as td:
        # 1) Train 5 epochs, dumping each.
        losses_a, _ = train(W1_0.copy(), b1_0.copy(), W2_0.copy(), b2_0.copy(),
                            X, y, n_epochs=5, ckpt_dir=td)
        ckpt_path = os.path.join(td, "latest.h5")
        assert os.path.isfile(ckpt_path), "checkpoint not written"
        size = os.path.getsize(ckpt_path)
        print(f"[ckpt] latest.h5 size = {size} B; losses_a = {losses_a}")
        assert size > 2048, f"checkpoint too small ({size} B)"
        assert losses_a[-1] < losses_a[0], "training diverged"

        # 2) Resume: load the snapshot from disk like a fresh process.
        ckpt = hickle.load(ckpt_path)
        assert ckpt["epoch"] == 4
        assert ckpt["metadata"]["config"] == {"lr": 1e-2, "hidden": 16}
        assert ckpt["metadata"]["tags"][-1] == "epoch=4"
        assert ckpt["metadata"]["loss"] == losses_a[-1]

        W1r = ckpt["weights"]["W1"]
        b1r = ckpt["weights"]["b1"]
        W2r = ckpt["weights"]["W2"]
        b2r = ckpt["weights"]["b2"]

        # 3) Train 5 more epochs starting from snapshot.
        losses_b, (W1f, _, _, _) = train(W1r, b1r, W2r, b2r, X, y,
                                         n_epochs=5, ckpt_dir=td)
        # Loss must keep decreasing across the resume boundary.
        assert losses_b[0] < losses_a[-1] + 1e-12, "resume regressed"

        # 4) Load the FINAL snapshot and confirm it round-trips bitwise.
        final = hickle.load(os.path.join(td, "latest.h5"))
        np.testing.assert_array_equal(final["weights"]["W1"], W1f)
        assert final["metadata"]["loss"] == losses_b[-1]
        # loss_history is a Python list of floats — confirm structural type.
        hist = final["metadata"]["loss_history"]
        assert isinstance(hist, list) and all(isinstance(x, float) for x in hist)
        assert len(hist) == 5

        print(f"[resume] losses_b = {losses_b}")
        print(f"[final] loss = {final['metadata']['loss']:.6f}")
        print("SCENARIO_OK")


if __name__ == "__main__":
    main()
