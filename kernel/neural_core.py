"""Neural core for the DRP Kernel: a self-learning decision graph.

This module provides a tiny, dependency-free neural layer that aligns with
the Decision Record Protocol (DRP). Every "neuron" is a DRP-compatible node
exposing ``neuron_id``, weighted ``inputs`` (causal trace), ``intent``
metadata (``intent_goal_id`` / ``intent_metrics``), and an
``outcome_metric`` measured after a decision is executed.

The graph supports a forward pass (threshold activation) and a simple
gradient-descent backward pass that updates input weights based on the
squared error between observed outcomes and both the factual target and the
declared intent. No external ML frameworks are required; only the Python
standard library is used.

Public API
----------
- :class:`KernelNeuron`
- :class:`KernelGraph`
- :func:`train_on_dataset`
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# KernelNeuron
# ---------------------------------------------------------------------------


class KernelNeuron:
    """A single DRP-compatible neuron in the decision graph.

    Parameters
    ----------
    neuron_id:
        Stable identifier of the neuron (analogous to DRP ``record_id``).
    inputs:
        Ordered list of input descriptors. Each entry is a dict with keys
        ``ref`` (id of the source neuron), ``weight`` (float, learnable),
        and ``signal`` (float, set during the forward pass).
    activation_threshold:
        Threshold used by the step activation function. Default ``0.5``.
    intent_goal_id:
        Optional identifier of the goal this neuron is aligned with.
    intent_metrics:
        Optional dict of expected target KPI values
        (e.g. ``{"target_revenue": 100}``).
    outcome_metric:
        Optional dict of metrics actually achieved after the decision was
        executed. Used by :meth:`compute_loss`.
    """

    def __init__(
        self,
        neuron_id: str,
        inputs: Optional[List[Dict[str, Any]]] = None,
        activation_threshold: float = 0.5,
        intent_goal_id: Optional[str] = None,
        intent_metrics: Optional[Dict[str, float]] = None,
        outcome_metric: Optional[Dict[str, float]] = None,
    ) -> None:
        self.neuron_id = neuron_id
        self.inputs: List[Dict[str, Any]] = list(inputs) if inputs else []
        self.activation_threshold = float(activation_threshold)
        self.intent_goal_id = intent_goal_id
        self.intent_metrics: Dict[str, float] = (
            dict(intent_metrics) if intent_metrics is not None else {}
        )
        self.outcome_metric: Optional[Dict[str, float]] = (
            dict(outcome_metric) if outcome_metric is not None else None
        )
        self.output: int = 0

    def forward(self) -> bool:
        """Compute the step-activated output and store it in ``self.output``.

        ``pre_activation`` is the dot product of weights and signals across
        all inputs. The neuron fires (``output = 1``) when the
        ``pre_activation`` reaches ``activation_threshold``; otherwise it is
        silent (``output = 0``). Sensor neurons with no inputs default to
        firing, since their signal is injected directly.

        Returns
        -------
        bool
            ``True`` if the neuron fired, ``False`` otherwise.
        """
        if not self.inputs:
            # Sensor / source neuron: it fires by default; its effective
            # signal is injected via ``KernelGraph.forward_pass``.
            self.output = 1
            return True

        pre_activation = sum(
            float(inp.get("weight", 0.0)) * float(inp.get("signal", 0.0))
            for inp in self.inputs
        )
        self.output = 1 if pre_activation >= self.activation_threshold else 0
        return bool(self.output)

    def compute_loss(self, target_outcome: Dict[str, float]) -> float:
        """Return the combined factual + intent squared error.

        ``factual_loss`` measures the gap between the observed
        ``outcome_metric`` and ``target_outcome``. ``goal_loss`` measures
        the gap between the declared ``intent_metrics`` and the same
        ``target_outcome``, capturing how well the original intent matched
        what the world actually demanded.

        If ``outcome_metric`` is not yet recorded, the loss is ``0.0``
        (the decision has not been executed and observed).
        """
        if self.outcome_metric is None:
            return 0.0

        factual_loss = 0.0
        for key, target_value in target_outcome.items():
            actual = float(self.outcome_metric.get(key, 0.0))
            factual_loss += (actual - float(target_value)) ** 2

        goal_loss = 0.0
        if self.intent_metrics:
            for key, target_value in target_outcome.items():
                expected = float(self.intent_metrics.get(key, 0.0))
                goal_loss += (expected - float(target_value)) ** 2

        return factual_loss + goal_loss

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of the neuron.

        Useful for debugging and for future bridging with the DRP record
        format. Field names mirror the DRP schema where possible.
        """
        return {
            "neuron_id": self.neuron_id,
            "inputs": [dict(inp) for inp in self.inputs],
            "activation_threshold": self.activation_threshold,
            "intent": {
                "goal_id": self.intent_goal_id,
                "metrics": dict(self.intent_metrics),
            },
            "outcome_metric": dict(self.outcome_metric) if self.outcome_metric else None,
            "output": self.output,
        }


# ---------------------------------------------------------------------------
# KernelGraph
# ---------------------------------------------------------------------------


class KernelGraph:
    """A directed graph of :class:`KernelNeuron` instances.

    The graph stores neurons in :attr:`neurons` and a forward adjacency map
    :attr:`adjacency` mapping ``ref -> [neuron_ids that consume ref]``,
    which allows the backward pass to walk from outputs back to sources.
    """

    def __init__(self) -> None:
        self.neurons: Dict[str, KernelNeuron] = {}
        # adjacency[ref] = list of neuron_ids that reference `ref` as input
        self.adjacency: Dict[str, List[str]] = {}

    # -- construction -------------------------------------------------------

    def add_neuron(
        self,
        neuron_id: str,
        inputs: Optional[List[Dict[str, Any]]] = None,
        threshold: float = 0.5,
        intent_goal_id: Optional[str] = None,
        intent_metrics: Optional[Dict[str, float]] = None,
    ) -> KernelNeuron:
        """Create a neuron, register it, and update the adjacency map.

        Returns the created :class:`KernelNeuron`.
        """
        if neuron_id in self.neurons:
            raise ValueError(f"neuron_id already registered: {neuron_id!r}")

        neuron = KernelNeuron(
            neuron_id=neuron_id,
            inputs=inputs,
            activation_threshold=threshold,
            intent_goal_id=intent_goal_id,
            intent_metrics=intent_metrics,
        )
        self.neurons[neuron_id] = neuron
        self.adjacency.setdefault(neuron_id, [])
        for inp in neuron.inputs:
            ref = inp.get("ref")
            if ref is None:
                continue
            self.adjacency.setdefault(ref, []).append(neuron_id)
        return neuron

    # -- forward pass -------------------------------------------------------

    def forward_pass(self, input_signals: Dict[str, float]) -> Dict[str, int]:
        """Propagate signals through the graph and return per-neuron outputs.

        For sensor neurons (no ``inputs``) the value from ``input_signals``
        is honoured: a sensor fires (``output = 1``) iff its injected
        signal is non-zero, and is silent otherwise. Sensor signals are
        also broadcast as the ``signal`` of any downstream input that
        references them.

        For non-sensor neurons, every input's ``signal`` is refreshed from
        ``input_signals`` (when present) before calling
        :meth:`KernelNeuron.forward`.
        """
        outputs: Dict[str, int] = {}

        # Pass 1: handle sensor neurons (no inputs) using injected signals.
        for neuron_id, neuron in self.neurons.items():
            if not neuron.inputs:
                signal = float(input_signals.get(neuron_id, 0.0))
                neuron.output = 1 if signal != 0.0 else 0
                outputs[neuron_id] = neuron.output

        # Pass 2: refresh downstream inputs and run their activations.
        for neuron_id, neuron in self.neurons.items():
            if not neuron.inputs:
                continue
            for inp in neuron.inputs:
                ref = inp.get("ref")
                if ref in input_signals:
                    inp["signal"] = float(input_signals[ref])
                elif ref in self.neurons and not self.neurons[ref].inputs:
                    # Source is a known sensor with no signal supplied;
                    # default its signal to 0.0 to keep behaviour explicit.
                    inp["signal"] = 0.0
            neuron.forward()
            outputs[neuron_id] = neuron.output

        return outputs

    # -- backward pass ------------------------------------------------------

    def backward_pass(
        self,
        neuron_id: str,
        loss_gradient: float,
        learning_rate: float = 0.01,
        _visited: Optional[set] = None,
    ) -> None:
        """Update input weights via gradient descent and recurse upstream.

        The activation is treated as a step function whose derivative is
        approximated as ``1`` so that learning still proceeds when the
        neuron fires. If the neuron is silent (``output == 0``), no update
        is applied and recursion stops, mirroring the fact that no causal
        contribution propagated through this branch.

        Parameters
        ----------
        neuron_id:
            Id of the neuron whose inputs should be updated.
        loss_gradient:
            Scalar gradient propagated from downstream (or the loss itself
            for the final neuron).
        learning_rate:
            Step size used by SGD.
        _visited:
            Internal cycle guard; do not pass explicitly.
        """
        if _visited is None:
            _visited = set()
        if neuron_id in _visited:
            return
        _visited.add(neuron_id)

        neuron = self.neurons.get(neuron_id)
        if neuron is None or neuron.output != 1:
            return

        # Derivative of the step activation w.r.t. pre-activation is taken
        # as 1 (a smooth-relaxation trick); gradient w.r.t. each weight is
        # therefore ``loss_gradient * 1 * signal``.
        for inp in neuron.inputs:
            signal = float(inp.get("signal", 0.0))
            gradient = loss_gradient * 1.0 * signal
            inp["weight"] = float(inp.get("weight", 0.0)) - learning_rate * gradient

            ref = inp.get("ref")
            if ref in self.neurons:
                self.backward_pass(
                    ref,
                    loss_gradient=loss_gradient,
                    learning_rate=learning_rate,
                    _visited=_visited,
                )

    # -- introspection ------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        """Return a JSON snapshot of every neuron in the graph."""
        return json.dumps(
            {nid: n.to_dict() for nid, n in self.neurons.items()},
            indent=indent,
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


Example = Tuple[Dict[str, float], Dict[str, Dict[str, float]]]


def _pre_activation(neuron: KernelNeuron) -> float:
    """Return the linear pre-activation (dot product) of a neuron."""
    return sum(
        float(inp.get("weight", 0.0)) * float(inp.get("signal", 0.0))
        for inp in neuron.inputs
    )


def train_on_dataset(
    graph: KernelGraph,
    dataset: Iterable[Example],
    epochs: int = 100,
    learning_rate: float = 0.01,
    verbose: bool = True,
) -> List[float]:
    """Train ``graph`` on a list of ``(input_signals, target_outcome)`` pairs.

    Training uses the continuous pre-activation as a smooth surrogate for
    the step-activated output (a straight-through-estimator approach), so
    that gradients can flow even when a neuron is silent at inference time.

    For every target neuron the routine:

    1. Computes the pre-activation ``z = Σ w_i · s_i``.
    2. Synthesises ``outcome_metric[k] = z · intent_metrics[k]`` for each
       ``k`` in the target — i.e. the neuron is "expected" to scale its
       intent metrics linearly with how confidently it fires.
    3. Records the squared error via :meth:`KernelNeuron.compute_loss`.
    4. Propagates a signed analytical gradient through
       :meth:`KernelGraph.backward_pass`. The neuron's output is forced
       to ``1`` for the duration of the backward call (STE), so weights
       can still move when the threshold has not yet been crossed.

    Parameters
    ----------
    graph:
        The :class:`KernelGraph` to train (mutated in place).
    dataset:
        Iterable of ``(input_signals, target_outcome)`` tuples. The target
        outcome is a mapping ``neuron_id -> {metric_key: value}``.
    epochs:
        Number of full passes over the dataset.
    learning_rate:
        SGD step size.
    verbose:
        When ``True`` (default), prints the average loss per epoch.

    Returns
    -------
    list[float]
        History of average losses, one entry per epoch.
    """
    samples: List[Example] = list(dataset)
    if not samples:
        raise ValueError("dataset must contain at least one example")

    loss_history: List[float] = []

    for epoch in range(epochs):
        total_loss = 0.0
        for input_signals, target_outcome in samples:
            graph.forward_pass(input_signals)

            example_loss = 0.0
            grads: Dict[str, float] = {}
            for neuron_id, target_metrics in target_outcome.items():
                neuron = graph.neurons.get(neuron_id)
                if neuron is None:
                    continue

                z = _pre_activation(neuron)

                # Synthesise outcome_metric from the continuous surrogate
                # so the loss is differentiable w.r.t. weights.
                synthesised: Dict[str, float] = {}
                grad = 0.0
                for key, target_value in target_metrics.items():
                    scale = float(neuron.intent_metrics.get(key, 1.0))
                    safe_scale = scale if scale != 0.0 else 1.0
                    predicted = z * scale
                    synthesised[key] = predicted
                    # Train in z-space: residual_z = z - target/scale.
                    # Gradient w.r.t. z is 2*residual_z so the learning
                    # rate stays well-conditioned regardless of metric
                    # magnitude (e.g. revenue in dollars vs. percent).
                    grad += 2.0 * (predicted - float(target_value)) / safe_scale
                neuron.outcome_metric = synthesised

                example_loss += neuron.compute_loss(target_metrics)
                grads[neuron_id] = grad

            for neuron_id, grad in grads.items():
                neuron = graph.neurons[neuron_id]
                saved_output = neuron.output
                neuron.output = 1  # STE: allow gradient flow during training
                try:
                    graph.backward_pass(
                        neuron_id,
                        loss_gradient=grad,
                        learning_rate=learning_rate,
                    )
                finally:
                    neuron.output = saved_output

            total_loss += example_loss

        avg_loss = total_loss / len(samples)
        loss_history.append(avg_loss)
        if verbose:
            print(f"epoch {epoch + 1:>4d}/{epochs}  avg_loss={avg_loss:.6f}")

    return loss_history


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------


def _build_demo_graph() -> KernelGraph:
    """Construct the canonical 3-sensor + 1-decision demo graph."""
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
    return graph


def _demo() -> None:
    """Run the documented demo: 4 examples, 100 epochs, lr=0.01."""
    graph = _build_demo_graph()

    dataset: List[Example] = [
        ({"sensor_a": 0.0, "sensor_b": 0.0, "sensor_c": 0.0},
         {"decision_1": {"target_revenue": 0}}),
        ({"sensor_a": 0.5, "sensor_b": 0.2, "sensor_c": 0.0},
         {"decision_1": {"target_revenue": 50}}),
        ({"sensor_a": 1.0, "sensor_b": 1.0, "sensor_c": 0.0},
         {"decision_1": {"target_revenue": 80}}),
        ({"sensor_a": 1.0, "sensor_b": 1.0, "sensor_c": 1.0},
         {"decision_1": {"target_revenue": 100}}),
    ]

    print("Initial decision_1 weights:")
    for inp in graph.neurons["decision_1"].inputs:
        print(f"  {inp['ref']}: {inp['weight']:.4f}")

    loss_history = train_on_dataset(
        graph, dataset, epochs=100, learning_rate=0.01, verbose=False
    )

    print("\nFinal decision_1 weights:")
    for inp in graph.neurons["decision_1"].inputs:
        print(f"  {inp['ref']}: {inp['weight']:.4f}")

    try:
        import matplotlib.pyplot as plt  # type: ignore

        plt.figure()
        plt.plot(range(1, len(loss_history) + 1), loss_history)
        plt.xlabel("epoch")
        plt.ylabel("avg loss")
        plt.title("DRP Kernel: training loss")
        plt.tight_layout()
        plt.savefig("kernel_loss.png")
        print("\nLoss curve saved to kernel_loss.png")
    except Exception:
        tail = loss_history[-5:]
        print("\nLast 5 epoch losses:")
        for i, value in enumerate(tail, start=len(loss_history) - len(tail) + 1):
            print(f"  epoch {i}: {value:.6f}")


if __name__ == "__main__":
    _demo()
