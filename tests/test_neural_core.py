"""Tests for the DRP Kernel neural core."""

from __future__ import annotations

import pytest

from kernel.neural_core import KernelGraph, KernelNeuron, train_on_dataset


# ---------------------------------------------------------------------------
# KernelNeuron
# ---------------------------------------------------------------------------


class TestKernelNeuron:
    def test_forward_fires_when_pre_activation_meets_threshold(self):
        neuron = KernelNeuron(
            "n1",
            inputs=[
                {"ref": "a", "weight": 0.6, "signal": 1.0},
                {"ref": "b", "weight": 0.0, "signal": 1.0},
            ],
            activation_threshold=0.5,
        )
        assert neuron.forward() is True
        assert neuron.output == 1

    def test_forward_silent_when_pre_activation_below_threshold(self):
        neuron = KernelNeuron(
            "n1",
            inputs=[{"ref": "a", "weight": 0.4, "signal": 1.0}],
            activation_threshold=0.5,
        )
        assert neuron.forward() is False
        assert neuron.output == 0

    def test_forward_sensor_neuron_fires_by_default(self):
        neuron = KernelNeuron("sensor", inputs=[])
        assert neuron.forward() is True
        assert neuron.output == 1

    def test_compute_loss_zero_when_outcome_missing(self):
        neuron = KernelNeuron("n1", intent_metrics={"x": 10})
        assert neuron.compute_loss({"x": 5}) == 0.0

    def test_compute_loss_factual_only_without_intent(self):
        neuron = KernelNeuron("n1", outcome_metric={"x": 7})
        # No intent_metrics → goal_loss is 0; only factual.
        assert neuron.compute_loss({"x": 10}) == pytest.approx(9.0)

    def test_compute_loss_factual_plus_goal(self):
        neuron = KernelNeuron(
            "n1",
            intent_metrics={"x": 10},
            outcome_metric={"x": 6},
        )
        # factual = (6-8)^2 = 4; goal = (10-8)^2 = 4; total = 8
        assert neuron.compute_loss({"x": 8}) == pytest.approx(8.0)

    def test_compute_loss_handles_missing_keys_in_outcome(self):
        neuron = KernelNeuron("n1", outcome_metric={})
        # Missing key defaults to 0 in outcome.
        assert neuron.compute_loss({"x": 5}) == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# KernelGraph
# ---------------------------------------------------------------------------


class TestKernelGraph:
    def test_add_neuron_registers_and_updates_adjacency(self):
        graph = KernelGraph()
        graph.add_neuron("a", inputs=[])
        graph.add_neuron(
            "b",
            inputs=[{"ref": "a", "weight": 1.0, "signal": 0.0}],
        )
        assert set(graph.neurons.keys()) == {"a", "b"}
        assert graph.adjacency["a"] == ["b"]

    def test_add_duplicate_neuron_raises(self):
        graph = KernelGraph()
        graph.add_neuron("a", inputs=[])
        with pytest.raises(ValueError):
            graph.add_neuron("a", inputs=[])

    def test_forward_pass_propagates_sensor_signals(self):
        graph = KernelGraph()
        graph.add_neuron("a", inputs=[])
        graph.add_neuron("b", inputs=[])
        graph.add_neuron(
            "out",
            inputs=[
                {"ref": "a", "weight": 0.5, "signal": 0.0},
                {"ref": "b", "weight": 0.5, "signal": 0.0},
            ],
            threshold=0.5,
        )
        outputs = graph.forward_pass({"a": 1.0, "b": 1.0})
        assert outputs["a"] == 1
        assert outputs["b"] == 1
        assert outputs["out"] == 1
        # Signals stored on inputs
        weights = {inp["ref"]: inp["signal"] for inp in graph.neurons["out"].inputs}
        assert weights == {"a": 1.0, "b": 1.0}

    def test_forward_pass_silent_below_threshold(self):
        graph = KernelGraph()
        graph.add_neuron("a", inputs=[])
        graph.add_neuron(
            "out",
            inputs=[{"ref": "a", "weight": 0.3, "signal": 0.0}],
            threshold=0.5,
        )
        outputs = graph.forward_pass({"a": 1.0})
        assert outputs["out"] == 0

    def test_backward_pass_skipped_when_neuron_silent(self):
        graph = KernelGraph()
        graph.add_neuron("a", inputs=[])
        graph.add_neuron(
            "out",
            inputs=[{"ref": "a", "weight": 0.3, "signal": 1.0}],
            threshold=0.5,
        )
        graph.forward_pass({"a": 1.0})
        assert graph.neurons["out"].output == 0
        original_weight = graph.neurons["out"].inputs[0]["weight"]
        graph.backward_pass("out", loss_gradient=10.0, learning_rate=0.1)
        assert graph.neurons["out"].inputs[0]["weight"] == original_weight

    def test_backward_pass_updates_weights_when_active(self):
        graph = KernelGraph()
        graph.add_neuron("a", inputs=[])
        graph.add_neuron(
            "out",
            inputs=[{"ref": "a", "weight": 0.6, "signal": 1.0}],
            threshold=0.5,
        )
        graph.forward_pass({"a": 1.0})
        assert graph.neurons["out"].output == 1
        graph.backward_pass("out", loss_gradient=1.0, learning_rate=0.1)
        # weight -= 0.1 * 1.0 * 1.0 = 0.1  →  0.6 - 0.1 = 0.5
        assert graph.neurons["out"].inputs[0]["weight"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


class TestTrainOnDataset:
    def _make_demo(self):
        graph = KernelGraph()
        graph.add_neuron("sensor_a", inputs=[])
        graph.add_neuron("sensor_b", inputs=[])
        graph.add_neuron("sensor_c", inputs=[])
        graph.add_neuron(
            "decision_1",
            inputs=[
                {"ref": "sensor_a", "weight": 0.2, "signal": 0.0},
                {"ref": "sensor_b", "weight": 0.2, "signal": 0.0},
                {"ref": "sensor_c", "weight": 0.2, "signal": 0.0},
            ],
            threshold=0.5,
            intent_goal_id="goal_1",
            intent_metrics={"target_revenue": 100},
        )
        dataset = [
            ({"sensor_a": 0.0, "sensor_b": 0.0, "sensor_c": 0.0},
             {"decision_1": {"target_revenue": 0}}),
            ({"sensor_a": 0.5, "sensor_b": 0.2, "sensor_c": 0.0},
             {"decision_1": {"target_revenue": 50}}),
            ({"sensor_a": 1.0, "sensor_b": 1.0, "sensor_c": 0.0},
             {"decision_1": {"target_revenue": 80}}),
            ({"sensor_a": 1.0, "sensor_b": 1.0, "sensor_c": 1.0},
             {"decision_1": {"target_revenue": 100}}),
        ]
        return graph, dataset

    def test_loss_history_length_matches_epochs(self):
        graph, dataset = self._make_demo()
        history = train_on_dataset(graph, dataset, epochs=10, learning_rate=0.01,
                                   verbose=False)
        assert len(history) == 10

    def test_loss_is_monotonically_non_increasing(self):
        graph, dataset = self._make_demo()
        history = train_on_dataset(graph, dataset, epochs=50, learning_rate=0.01,
                                   verbose=False)
        # SGD with full-batch-like accumulation should not increase here.
        for prev, curr in zip(history, history[1:]):
            assert curr <= prev + 1e-9

    def test_demo_converges_to_least_squares_solution(self):
        graph, dataset = self._make_demo()
        train_on_dataset(graph, dataset, epochs=2000, learning_rate=0.05,
                         verbose=False)
        weights = {inp["ref"]: inp["weight"]
                   for inp in graph.neurons["decision_1"].inputs}
        # Analytical LSQ optimum with 4 examples / 3 unknowns:
        # w_a ≈ 1.133, w_b ≈ -0.333, w_c = 0.2
        assert weights["sensor_a"] == pytest.approx(1.1333, abs=0.05)
        assert weights["sensor_b"] == pytest.approx(-0.3333, abs=0.05)
        assert weights["sensor_c"] == pytest.approx(0.2000, abs=0.05)

    def test_empty_dataset_raises(self):
        graph, _ = self._make_demo()
        with pytest.raises(ValueError):
            train_on_dataset(graph, [], epochs=10, verbose=False)
