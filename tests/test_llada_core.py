"""CPU-only checks for the released LLaDA search/decoding interface."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llada_cal"))
from llada_cal import generate, get_search_confidence, load_bias_params


class ToyModel:
    def __init__(self, preferred_length=3):
        self.device = torch.device("cpu")
        self.preferred_length = preferred_length
        self.calls = []

    def __call__(self, tokens, attention_mask=None):
        self.calls.append(tokens.clone())
        length = tokens.shape[1] - 3
        logits = torch.zeros(*tokens.shape, 32)
        logits[..., 4] = 6.0 - 0.5 * abs(length - self.preferred_length)
        return SimpleNamespace(logits=logits)


class CALCoreTests(unittest.TestCase):
    def run_generation(self, model, **overrides):
        options = dict(
            prefix_ids=torch.tensor([[1, 2]]), suffix_ids=torch.tensor([[3]]),
            gen_length=4, block_length=4, span=1, dstep=2,
            max_gen_length=10, mask_id=31, use_bias=True,
            bias_params=[0.0, 1.0, 0.0, 1.0, 1.0],
        )
        options.update(overrides)
        return generate(model, **options)

    def test_selected_canvas_is_complete_and_context_is_fixed(self):
        for length in (2, 5, 7):
            model = ToyModel(length)
            output, search_steps, search_time, decode_time = self.run_generation(model)
            self.assertEqual(output.shape[1], length + 3)
            self.assertFalse((output == 31).any())
            self.assertEqual(len(model.calls), search_steps + length)
            self.assertGreaterEqual(search_time, 0)
            self.assertGreaterEqual(decode_time, 0)
            for tokens in model.calls:
                self.assertEqual(tokens[0, :2].tolist(), [1, 2])
                self.assertEqual(tokens[0, -1].item(), 3)

    def test_bidirectional_probe_order(self):
        model = ToyModel(2)
        _, probes, _, _ = self.run_generation(model)
        lengths = [tokens.shape[1] - 3 for tokens in model.calls[:probes]]
        self.assertEqual(lengths, [4, 5, 6, 3, 2, 1])

    def test_explicit_steps_are_not_replaced_by_selected_length(self):
        model = ToyModel(2)
        output, probes, _, _ = self.run_generation(model, steps=16)
        self.assertEqual(output.shape[1], 5)
        self.assertEqual(len(model.calls), probes + 16)
        self.assertFalse((output == 31).any())

    def test_fixed_length_has_no_search(self):
        model = ToyModel()
        output, probes, search_time, _ = self.run_generation(model, dstep=-1, steps=2)
        self.assertEqual(probes, 0)
        self.assertEqual(search_time, 0)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(output.shape[1], 7)
        self.assertFalse((output == 31).any())

    def test_alternative_scores_change_search_only(self):
        for score in ("margin", "entropy"):
            model = ToyModel(5)
            output, probes, _, _ = self.run_generation(model, search_score=score)
            self.assertEqual(output[0, 2:-1].tolist(), [4] * 5)
            self.assertEqual(len(model.calls), probes + 5)

    def test_scores_match_definitions(self):
        logits = torch.tensor([[[1.0, 2.0, 3.0]]])
        probabilities = logits.softmax(-1)
        torch.testing.assert_close(get_search_confidence(logits, "top1"), probabilities.max(-1).values)
        top2 = probabilities.topk(2, dim=-1).values
        torch.testing.assert_close(get_search_confidence(logits, "margin"), top2[..., 0] - top2[..., 1])
        expected = 1 + (probabilities * probabilities.log()).sum(-1) / torch.log(torch.tensor(3.0))
        torch.testing.assert_close(get_search_confidence(logits, "entropy"), expected)

    def test_frozen_bias_loads(self):
        directory = Path(__file__).resolve().parents[1] / "motivation"
        for name in ("bias_top1_fitted.json", "bias_margin.json", "bias_entropy.json", "bias_params_roc.json"):
            self.assertEqual(len(load_bias_params(directory / name)), 5)


if __name__ == "__main__":
    unittest.main()
