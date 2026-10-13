#!/usr/bin/env python3
"""
Tests for `scripts/pth2onnx_ismcts.py`.

The export path is the last place where a mistake is still cheap: everything downstream
(the Go backend) just runs whatever graph it is given, and a value head exported under the
wrong target keeps producing legal actions forever — only the numbers mean something else.
So these tests pin the guards rather than the export itself:

  1. a checkpoint with no provenance sidecar and no `--value-target` is REFUSED (the two
     heads are the same weights with a different activation, so guessing is unsound),
  2. a sidecar's `value_target` is used when the flag is absent,
  3. the exported graph's head activation is checked against that claim: the same weights
     exported as `win` and as `return` produce graphs whose `value` paths differ, and
     claiming the wrong one is an error,
  4. the export round-trips: torch and ONNX agree on the logits, the value and the argmax.

Run: python3 scripts/test_pth2onnx_ismcts.py
"""

import json
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src" / "ismcts"))
sys.path.insert(0, str(REPO / "scripts"))

from network import CuarentaNet                       # noqa: E402
import pth2onnx_ismcts as exporter                    # noqa: E402


def _tiny_net(value_target: str) -> CuarentaNet:
    torch.manual_seed(0)
    net = CuarentaNet(hidden=32, lstm_layers=1, value_target=value_target)
    net.eval()
    return net


def test_refuses_to_guess_without_a_sidecar():
    """No flag + no sidecar must be an error, not a default."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "c.onnx.pt"
        torch.save(_tiny_net("return").state_dict(), ckpt)
        try:
            exporter.resolve_value_target(ckpt, None)
        except SystemExit as exc:
            assert "--value-target" in str(exc), str(exc)
            print("test_refuses_to_guess_without_a_sidecar: OK (refused, and said how)")
            return
        raise AssertionError("resolve_value_target guessed instead of refusing")


def test_flag_and_sidecar_both_work():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "c.pt"
        torch.save(_tiny_net("return").state_dict(), ckpt)

        target, _meta, source = exporter.resolve_value_target(ckpt, "win")
        assert (target, source) == ("win", "flag"), (target, source)

        # The trainer's own convention: <checkpoint>.json next to the weights.
        Path(str(ckpt) + ".json").write_text(json.dumps({"value_target": "return"}))
        target, meta, source = exporter.resolve_value_target(ckpt, None)
        assert (target, source) == ("return", "sidecar"), (target, source)
        assert meta["value_target"] == "return"

        # An explicit flag still wins over the sidecar (and is reported as such).
        target, _meta, source = exporter.resolve_value_target(ckpt, "win")
        assert (target, source) == ("win", "flag"), (target, source)
        print("test_flag_and_sidecar_both_work: OK")


def test_head_activation_is_checked_against_the_claim():
    """The same weights exported twice must produce different `value` graphs, and claiming
    the wrong activation for either graph must fail."""
    import onnx

    with tempfile.TemporaryDirectory() as tmp:
        win_ckpt = Path(tmp) / "win.pt"
        ret_ckpt = Path(tmp) / "ret.pt"
        torch.save(_tiny_net("win").state_dict(), win_ckpt)
        torch.save(_tiny_net("return").state_dict(), ret_ckpt)

        win_onnx = exporter.export(win_ckpt, Path(tmp) / "win.onnx", value_target="win",
                                   verify=False)
        ret_onnx = exporter.export(ret_ckpt, Path(tmp) / "ret.onnx", value_target="return",
                                   verify=False)

        # Right claims pass.
        exporter.check_head_activation(onnx.load(str(win_onnx)), "win")
        exporter.check_head_activation(onnx.load(str(ret_onnx)), "return")

        # Wrong claims fail.
        for model_path, wrong in ((win_onnx, "return"), (ret_onnx, "win")):
            try:
                exporter.check_head_activation(onnx.load(str(model_path)), wrong)
            except SystemExit as exc:
                assert "Tanh" in str(exc), str(exc)
            else:
                raise AssertionError(f"claimed {wrong} for {model_path.name} and it passed")
        print("test_head_activation_is_checked_against_the_claim: OK")


def test_export_round_trips_through_onnx():
    """The full path including the built-in parity check (torch vs ONNX on real decisions)."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ret.pt"
        net = _tiny_net("return")
        torch.save(net.state_dict(), ckpt)
        onnx_path = exporter.export(ckpt, Path(tmp) / "ret.onnx", value_target="return",
                                    verify=True)
        meta = json.loads(Path(str(onnx_path).replace(".onnx", ".json")).read_text())
        assert meta["value_target"] == "return"
        assert meta["recurrent_state"] is None
        assert "NOT bounded by 1" in meta["value_semantics"], meta["value_semantics"]
        assert meta["value_target_source"] == "flag"
        print("test_export_round_trips_through_onnx: OK")


if __name__ == "__main__":
    test_refuses_to_guess_without_a_sidecar()
    test_flag_and_sidecar_both_work()
    test_head_activation_is_checked_against_the_claim()
    test_export_round_trips_through_onnx()
    print("All ISMCTS ONNX export tests passed.")
