"""End-to-end smoke test on tiny synthetic data (no MVSEC files needed).

Checks: pretrain TBPTT forward/backward, depth forward/backward, baseline,
frozen-probe gradient isolation, and streaming-vs-window consistency.

    python tests/smoke_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.cells import detach_states
from src.models import SpikingDepthModel, SpikingFutureModel, UNetDepthBaseline

B, T, K, H, W = 2, 8, 3, 64, 80
CH = (8, 16, 24, 32)


def test_pretrain_tbptt():
    model = SpikingFutureModel(stage_channels=CH, fused_channels=16, pred_bins=K)
    bins = (torch.rand(B, T + K, 2, H, W) * 3).floor()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    states = None
    opt.zero_grad()
    chunk_losses = []
    for t0 in range(0, T, 4):
        loss, metrics, states = model.forward_chunk(bins, t0, min(t0 + 4, T), states, warmup=2)
        if loss is not None:
            loss.backward()
            chunk_losses.append(float(loss.detach()))
        states = detach_states(states)
    opt.step()
    assert chunk_losses, "no loss produced"
    assert all(torch.isfinite(torch.tensor(chunk_losses))), chunk_losses
    assert 0.0 <= metrics["f1"] <= 1.0
    print(f"pretrain TBPTT ok: chunk losses {[f'{l:.4f}' for l in chunk_losses]}, f1 {metrics['f1']:.3f}")


def test_depth_and_streaming():
    model = SpikingDepthModel(stage_channels=CH, fused_channels=16, depth_min=0.5, depth_max=10.0)
    bins = (torch.rand(B, T, 2, H, W) * 3).floor()

    depth, states = model(bins)
    assert depth.shape == (B, H, W)
    assert float(depth.min()) >= 0.5 and float(depth.max()) <= 10.0

    from src.losses import silog_loss
    target = torch.rand(B, H, W) * 9 + 0.6
    valid = torch.rand(B, H, W) > 0.3
    loss = silog_loss(depth, target, valid)
    loss.backward()
    assert torch.isfinite(loss)

    # streaming step continues from the window's state and stays in bounds
    model.eval()
    with torch.no_grad():
        depth2, states = model.step(bins[:, -1], detach_states(states))
    assert depth2.shape == (B, H, W)

    # window forward == manual step-by-step (same state evolution)
    model2 = SpikingDepthModel(stage_channels=CH, fused_channels=16).eval()
    with torch.no_grad():
        d_window, _ = model2(bins)
        st = None
        for t in range(T):
            d_steps, st = model2.step(bins[:, t], st)
    assert torch.allclose(d_window, d_steps, atol=1e-5), "streaming and window paths diverge"
    print(f"depth ok: range [{float(depth.min()):.2f}, {float(depth.max()):.2f}], "
          f"streaming==window max diff {float((d_window - d_steps).abs().max()):.2e}")


def test_frozen_probe():
    model = SpikingDepthModel(stage_channels=CH, fused_channels=16, freeze_encoder=True)
    bins = (torch.rand(B, T, 2, H, W) * 3).floor()
    depth, _ = model(bins)
    depth.mean().backward()
    assert all(p.grad is None for p in model.encoder.parameters()), "frozen encoder got grads"
    assert any(p.grad is not None for p in model.depth_head.parameters()), "head got no grads"
    print("frozen probe ok: encoder isolated, head trains")


def test_streaming_wrapper():
    from src.streaming import StreamingDepthNet

    model = SpikingDepthModel(stage_channels=CH, fused_channels=16).eval()
    net = StreamingDepthNet(model).eval()
    bins = (torch.rand(B, T, 2, H, W) * 3).floor()

    d_ref = d_wrap = torch.empty(0)
    with torch.no_grad():
        # reference: the training-path model, step by step
        st = None
        for t in range(T):
            d_ref, st = model.step(bins[:, t], st)
        # wrapper: same bins through flat-tensor state
        state = net.init_state(B, H, W)
        for t in range(T):
            out = net(bins[:, t], *state)
            d_wrap, state = out[0], out[1:]
    assert torch.allclose(d_ref, d_wrap, atol=1e-5), "StreamingDepthNet diverges from model.step"
    assert len(state) == 2 * net.n_stages
    print(f"streaming wrapper ok: max diff {float((d_ref - d_wrap).abs().max()):.2e}, "
          f"{len(state)} state tensors")


def test_baseline():
    model = UNetDepthBaseline(input_bins=T, channels=CH, depth_min=0.5, depth_max=10.0)
    bins = (torch.rand(B, T, 2, H, W) * 3).floor()
    depth = model(bins)
    assert depth.shape == (B, H, W)
    depth.mean().backward()
    print("baseline U-Net ok")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_pretrain_tbptt()
    test_depth_and_streaming()
    test_frozen_probe()
    test_streaming_wrapper()
    test_baseline()
    print("\nall smoke tests passed")
