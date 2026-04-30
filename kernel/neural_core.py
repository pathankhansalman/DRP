"""Neural core for the DRP Kernel: a self-learning decision graph.

This module provides a tiny, dependency-free neural layer that aligns with
the Decision Record Protocol (DRP). Every "neuron" is a DRP-compatible node
exposing ``neuron_id``, weighted ``inputs`` (causal trace), ``intent``
metadata (``intent_goal_id`` / ``intent_metrics``), and an
``outcome_metric`` measured after a decision is executed.

The graph supports a forward pass (threshold activation in topological
order) and a simple gradient-descent backward pass that updates input
weights based on the squared error between observed outcomes and both the
factual target and the declared intent. No external ML frameworks are
required; only the Python standard library is used.

Public API
----------
- :class:`KernelNeuron`
- :class:`KernelGraph`
- :func:`train_on_dataset`
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, TypedDict


# Straight-through-estimator: the derivative of the step activation w.r.t.
# its pre-activation is approximated as 1 so that a learning signal can
# propagate through a non-differentiable threshold.
_STEP_DERIVATIVE = 1.0


class Synapse(TypedDict, total=False):
    """One weighted, causal connection from a source neuron.

    ``ref`` is the id of the source. ``weight`` is the learnable scalar.
    ``signal`` is the most recent value broadcast by the source during a
    forward pass. ``total=False`` keeps the dict permissive so existing
    DRP fixtures with extra fields can still be loaded.
    """

    ref: str
    weight: float
    signal: float


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
        Ordered list of :class:`Synapse` descriptors. Each entry has keys
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
        inputs: Optional[List[Synapse]] = None,
        activation_threshold: float = 0.5,
        intent_goal_id: Optional[str] = None,
        intent_metrics: Optional[Dict[str, float]] = None,
        outcome_metric: Optional[Dict[str, float]] = None,
    ) -> None:
        self.neuron_id = neuron_id
        self.inputs: List[Synapse] = [dict(inp) for inp in inputs] if inputs else []
        self.activation_threshold = float(activation_threshold)
        self.intent_goal_id = intent_goal_id
        self.intent_metrics: Dict[str, float] = (
            dict(intent_metrics) if intent_metrics is not None else {}
        )
        self.outcome_metric: Optional[Dict[str, float]] = (
            dict(outcome_metric) if outcome_metric is not None else None
        )
        self.output: int = 0

    @property
    def is_sensor(self) -> bool:
        """Return ``True`` for source nodes that have no upstream inputs."""
        return not self.inputs

    def pre_activation(self) -> float:
        """Return the dot product ``Σ weight · signal`` over all inputs."""
        return sum(
            float(inp.get("weight", 0.0)) * float(inp.get("signal", 0.0))
            for inp in self.inputs
        )

    def forward(self) -> bool:
        """Compute the step-activated output and store it in ``self.output``.

        Sensor neurons (no inputs) keep whatever output was injected by
        :meth:`KernelGraph.forward_pass`; calling ``forward`` on them is a
        no-op so the two code paths agree on sensor activation.

        Returns
        -------
        bool
            ``True`` if the neuron fires after this call, else ``False``.
        """
        if self.is_sensor:
            return bool(self.output)

        self.output = 1 if self.pre_activation() >= self.activation_threshold else 0
        return bool(self.output)

    def compute_loss(self, target_outcome: Dict[str, float]) -> float:
        """Return the combined factual + intent squared error.

        ``factual_loss`` measures the gap between the observed
        ``outcome_metric`` and ``target_outcome``. ``goal_loss`` measures
        the gap between the declared ``intent_metrics`` and the same
        ``target_outcome``, capturing how well the original intent matched
        what the world actually demanded.

        Returns ``0.0`` when ``outcome_metric`` is ``None`` — i.e. the
        decision has not been executed and observed yet.
        """
        if self.outcome_metric is None:
            return 0.0

        factual_loss = sum(
            (float(self.outcome_metric.get(key, 0.0)) - float(target_value)) ** 2
            for key, target_value in target_outcome.items()
        )
        if not self.intent_metrics:
            return factual_loss

        goal_loss = sum(
            (float(self.intent_metrics.get(key, 0.0)) - float(target_value)) ** 2
            for key, target_value in target_outcome.items()
        )
        return factual_loss + goal_loss

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot mirroring the DRP schema."""
        return {
            "neuron_id": self.neuron_id,
            "inputs": [dict(inp) for inp in self.inputs],
            "activation_threshold": self.activation_threshold,
            "intent": {
                "goal_id": self.intent_goal_id,
                "metrics": dict(self.intent_metrics),
            },
            "outcome_metric": (
                dict(self.outcome_metric) if self.outcome_metric is not None else None
            ),
            "output": self.output,
        }

    def __repr__(self) -> str:
        return (
            f"KernelNeuron(id={self.neuron_id!r}, "
            f"inputs={len(self.inputs)}, "
            f"threshold={self.activation_threshold}, "
            f"output={self.output})"
        )


# ---------------------------------------------------------------------------
# KernelGraph
# ---------------------------------------------------------------------------


class KernelGraph:
    """A directed graph of :class:`KernelNeuron` instances.

    The graph stores neurons in :attr:`neurons` and a forward adjacency map
    :attr:`adjacency` mapping ``ref -> [neuron_ids that consume ref]``,
    which lets the backward pass walk from outputs back to sources.
    """

    def __init__(self) -> None:
        self.neurons: Dict[str, KernelNeuron] = {}
        # adjacency[ref] = list of neuron_ids that reference `ref` as input
        self.adjacency: Dict[str, List[str]] = {}

    # -- construction -------------------------------------------------------

    def add_neuron(
        self,
        neuron_id: str,
        inputs: Optional[List[Synapse]] = None,
        threshold: float = 0.5,
        intent_goal_id: Optional[str] = None,
        intent_metrics: Optional[Dict[str, float]] = None,
    ) -> KernelNeuron:
        """Create a neuron, register it, and update the adjacency map.

        Returns the created :class:`KernelNeuron`. Raises ``ValueError`` if
        ``neuron_id`` is already registered.
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

    def topological_order(self) -> List[str]:
        """Return neuron ids in dependency order (sources first).

        Uses Kahn's algorithm over the input edges. Cyclic graphs raise
        ``ValueError``: the DRP causal graph is required to be acyclic.
        """
        in_degree: Dict[str, int] = {nid: 0 for nid in self.neurons}
        for nid, neuron in self.neurons.items():
            for inp in neuron.inputs:
                ref = inp.get("ref")
                if ref in self.neurons:
                    in_degree[nid] += 1

        ready = [nid for nid, deg in in_degree.items() if deg == 0]
        order: List[str] = []
        while ready:
            nid = ready.pop(0)
            order.append(nid)
            for downstream in self.adjacency.get(nid, []):
                in_degree[downstream] -= 1
                if in_degree[downstream] == 0:
                    ready.append(downstream)

        if len(order) != len(self.neurons):
            raise ValueError("KernelGraph contains a cycle")
        return order

    def forward_pass(self, input_signals: Dict[str, float]) -> Dict[str, int]:
        """Propagate signals through the graph and return per-neuron outputs.

        Sensors take their output from ``input_signals`` directly: a sensor
        fires (``output = 1``) iff its injected signal is non-zero. The
        emitted "signal" of every neuron — sensor or not — is its
        pre-activation (or the injected raw signal for sensors), which is
        then broadcast to downstream synapses before they are activated.

        Neurons are visited in topological order so multi-layer graphs
        propagate correctly in a single pass.
        """
        outputs: Dict[str, int] = {}
        emitted_signal: Dict[str, float] = {}

        for nid in self.topological_order():
            neuron = self.neurons[nid]

            if neuron.is_sensor:
                signal = float(input_signals.get(nid, 0.0))
                neuron.output = 1 if signal != 0.0 else 0
                emitted_signal[nid] = signal
            else:
                for inp in neuron.inputs:
                    ref = inp.get("ref")
                    if ref in input_signals:
                        inp["signal"] = float(input_signals[ref])
                    elif ref in emitted_signal:
                        inp["signal"] = emitted_signal[ref]
                neuron.forward()
                emitted_signal[nid] = neuron.pre_activation()

            outputs[nid] = neuron.output

        return outputs

    # -- backward pass ------------------------------------------------------

    def backward_pass(
        self,
        neuron_id: str,
        loss_gradient: float,
        learning_rate: float = 0.01,
        force: bool = False,
        _visited: Optional[set] = None,
    ) -> None:
        """Update input weights via gradient descent and recurse upstream.

        The activation is treated as a step function whose derivative is
        approximated as ``1`` (straight-through estimator). When the
        neuron is silent (``output == 0``) and ``force`` is ``False`` no
        update is applied and recursion stops, mirroring the fact that no
        causal contribution propagated through this branch.

        Parameters
        ----------
        neuron_id:
            Id of the neuron whose inputs should be updated.
        loss_gradient:
            Scalar gradient propagated from downstream (or the loss-derived
            gradient for the final neuron).
        learning_rate:
            Step size used by SGD.
        force:
            When ``True``, update weights even if the neuron did not fire.
            Used by :func:`train_on_dataset` so silent neurons can still
            move toward firing.
        _visited:
            Internal cycle / re-entry guard. Do not pass explicitly.
        """
        if _visited is None:
            _visited = set()
        if neuron_id in _visited:
            return
        _visited.add(neuron_id)

        neuron = self.neurons.get(neuron_id)
        if neuron is None:
            return
        if not force and neuron.output != 1:
            return

        for inp in neuron.inputs:
            signal = float(inp.get("signal", 0.0))
            gradient = loss_gradient * _STEP_DERIVATIVE * signal
            inp["weight"] = float(inp.get("weight", 0.0)) - learning_rate * gradient

            ref = inp.get("ref")
            if ref in self.neurons:
                # Upstream recursion never forces — only neurons that
                # actually contributed a positive signal get updated.
                self.backward_pass(
                    ref,
                    loss_gradient=loss_gradient,
                    learning_rate=learning_rate,
                    force=False,
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

    def __repr__(self) -> str:
        return f"KernelGraph(neurons={len(self.neurons)})"


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


Example = tuple[Dict[str, float], Dict[str, Dict[str, float]]]


def _synthesise_outcome_and_grad(
    neuron: KernelNeuron, target_metrics: Dict[str, float]
) -> tuple[Dict[str, float], float]:
    """Synthesise an outcome from the pre-activation and return its grad.

    The neuron is "expected" to scale its intent metrics linearly with how
    confidently it fires: ``outcome_metric[k] = z · intent_metrics[k]``,
    where ``z`` is the pre-activation. Training therefore happens in
    z-space, which keeps the learning rate well-conditioned regardless of
    whether the metric is measured in dollars, percent, or seconds.
    """
    z = neuron.pre_activation()
    synthesised: Dict[str, float] = {}
    grad = 0.0
    for key, target_value in target_metrics.items():
        scale = float(neuron.intent_metrics.get(key, 1.0))
        safe_scale = scale if scale != 0.0 else 1.0
        predicted = z * scale
        synthesised[key] = predicted
        # d/dz (z·scale - target)^2 = 2·(predicted - target)·scale.
        # Dividing by `scale` once normalises the gradient back to z-space.
        grad += 2.0 * (predicted - float(target_value)) / safe_scale
    return synthesised, grad


def _train_example(
    graph: KernelGraph,
    input_signals: Dict[str, float],
    target_outcome: Dict[str, Dict[str, float]],
    learning_rate: float,
) -> float:
    """Run forward + backward for a single (signals, targets) pair.

    Returns the squared-error loss summed over all target neurons.
    """
    graph.forward_pass(input_signals)

    example_loss = 0.0
    grads: Dict[str, float] = {}
    for neuron_id, target_metrics in target_outcome.items():
        neuron = graph.neurons.get(neuron_id)
        if neuron is None:
            continue
        synthesised, grad = _synthesise_outcome_and_grad(neuron, target_metrics)
        neuron.outcome_metric = synthesised
        example_loss += neuron.compute_loss(target_metrics)
        grads[neuron_id] = grad

    for neuron_id, grad in grads.items():
        graph.backward_pass(
            neuron_id,
            loss_gradient=grad,
            learning_rate=learning_rate,
            force=True,
        )
    return example_loss


def train_on_dataset(
    graph: KernelGraph,
    dataset: Iterable[Example],
    epochs: int = 100,
    learning_rate: float = 0.01,
    verbose: bool = True,
) -> List[float]:
    """Train ``graph`` on a list of ``(input_signals, target_outcome)`` pairs.

    Training uses the continuous pre-activation as a smooth surrogate for
    the step-activated output (a straight-through estimator), so gradients
    can flow even when a neuron is silent at inference time. For each
    target neuron, the routine synthesises ``outcome_metric`` from its
    pre-activation, records the squared error, and propagates a signed
    analytical gradient through :meth:`KernelGraph.backward_pass`.

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
            total_loss += _train_example(
                graph, input_signals, target_outcome, learning_rate
            )

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


def _demo_dataset() -> List[Example]:
    return [
        ({"sensor_a": 0.0, "sensor_b": 0.0, "sensor_c": 0.0},
         {"decision_1": {"target_revenue": 0}}),
        ({"sensor_a": 0.5, "sensor_b": 0.2, "sensor_c": 0.0},
         {"decision_1": {"target_revenue": 50}}),
        ({"sensor_a": 1.0, "sensor_b": 1.0, "sensor_c": 0.0},
         {"decision_1": {"target_revenue": 80}}),
        ({"sensor_a": 1.0, "sensor_b": 1.0, "sensor_c": 1.0},
         {"decision_1": {"target_revenue": 100}}),
    ]


def _print_weights(label: str, neuron: KernelNeuron) -> None:
    print(f"{label}:")
    for inp in neuron.inputs:
        print(f"  {inp['ref']}: {inp['weight']:.4f}")


def _plot_or_tail_losses(loss_history: List[float]) -> None:
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


def _demo() -> None:
    """Run the documented demo: 4 examples, 100 epochs, lr=0.01."""
    graph = _build_demo_graph()
    dataset = _demo_dataset()

    decision = graph.neurons["decision_1"]
    _print_weights("Initial decision_1 weights", decision)

    loss_history = train_on_dataset(
        graph, dataset, epochs=100, learning_rate=0.01, verbose=False
    )

    print()
    _print_weights("Final decision_1 weights", decision)
    _plot_or_tail_losses(loss_history)


if __name__ == "__main__":
    _demo()
