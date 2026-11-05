# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Accumulate temporal example assertions without per-step host readback."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton


@wp.struct
class _Result:
    observed: int
    step: int
    index: int
    time: wp.float64
    extent: float


@wp.kernel(enable_backward=False)
def _reset(failures: wp.array[int], bounds: wp.array2d[float]):
    i = wp.tid()
    failures[i] = 2147483647
    for axis in range(3):
        bounds[i, axis] = wp.inf
        bounds[i, axis + 3] = -wp.inf


@wp.kernel(enable_backward=False)
def _finish(
    failures: wp.array[int],
    bounds: wp.array2d[float],
    start_times: wp.array[wp.float64],
    extent_limits: wp.array[float],
    step: int,
    sim_time: wp.float64,
    results: wp.array[_Result],
):
    i = wp.tid()
    if sim_time < start_times[i]:
        return
    result = results[i]
    result.observed += 1
    if result.step == 0:
        extent = wp.vec3(bounds[i, 3] - bounds[i, 0], bounds[i, 4] - bounds[i, 1], bounds[i, 5] - bounds[i, 2])
        if failures[i] != 2147483647:
            result.step = step
            result.index = failures[i]
            result.time = sim_time
        elif extent_limits[i] > 0.0 and not wp.length(extent) < extent_limits[i]:
            result.step = step
            result.index = -1
            result.time = sim_time
            result.extent = wp.length(extent)
    results[i] = result


def _body_kernel(test_fn):
    @wp.kernel(module="unique", enable_backward=False)
    def check(
        q: wp.array[wp.transform],
        qd: wp.array[wp.spatial_vector],
        indices: wp.array[int],
        check_id: int,
        failures: wp.array[int],
    ):
        index = indices[wp.tid()]
        if not test_fn(q[index], qd[index]):
            wp.atomic_min(failures, check_id, index)

    return check


def _particle_kernel(test_fn):
    @wp.kernel(module="unique", enable_backward=False)
    def check(
        q: wp.array[wp.vec3],
        qd: wp.array[wp.vec3],
        indices: wp.array[int],
        check_id: int,
        check_extent: bool,
        failures: wp.array[int],
        bounds: wp.array2d[float],
    ):
        index = indices[wp.tid()]
        pos = q[index]
        if not test_fn(pos, qd[index]):
            wp.atomic_min(failures, check_id, index)
        if check_extent:
            for axis in range(3):
                wp.atomic_min(bounds, check_id, axis, pos[axis])
                wp.atomic_max(bounds, check_id, axis + 3, pos[axis])

    return check


@dataclass
class _Check:
    name: str
    kind: str
    kernel: wp.Kernel
    indices: wp.array[int]
    start_time: float
    max_extent: float | None


class ExampleTestMonitor:
    """Retain temporal body and particle check failures on the simulation device.

    .. experimental::

    Register checks before the first :meth:`observe` call. Call ``observe`` from
    an example's ``test_post_step()`` and :meth:`assert_valid` from its
    ``test_final()``. Observation launches Warp kernels without host readback;
    final validation copies one fixed-size record per check to the host. Failure
    records survive subsequent recovery of the simulation state.

    Create the monitor only in test mode. Observations run outside the simulation
    CUDA graph, on the current stream for the model device. A monitor belongs to
    one example run; create a new one when resetting the example. It does not
    retain full trajectories or replace example-specific final-state assertions.

    Args:
        model: Model defining the body and particle indices and device.
    """

    def __init__(self, model: newton.Model):
        self._model = model
        self._checks: list[_Check] = []
        self._results = None
        self._step = 0

    def add_body_check(
        self,
        name: str,
        test_fn: wp.Function | Callable[[wp.transform, wp.spatial_vector], bool],
        *,
        indices: Sequence[int] | None = None,
        start_time: float = 0.0,
    ) -> None:
        """Register a predicate evaluated for each selected body's pose and twist.

        Args:
            name: Unique, descriptive check name used in failure messages.
            test_fn: Warp-compatible predicate accepting pose [m, quaternion]
                and twist [rad/s, m/s], returning True for an acceptable state.
            indices: Body indices, or None to check all bodies. Must be nonempty.
            start_time: First simulation time [s] at which the check is required.
                Final validation fails if no observation reaches this time.
        """
        self._add_check(name, test_fn, "body", indices, start_time, None)

    def add_particle_check(
        self,
        name: str,
        test_fn: wp.Function | Callable[[wp.vec3, wp.vec3], bool],
        *,
        indices: Sequence[int] | None = None,
        start_time: float = 0.0,
        max_extent: float | None = None,
    ) -> None:
        """Register a particle predicate and an optional subset extent limit.

        Args:
            name: Unique, descriptive check name used in failure messages.
            test_fn: Warp-compatible predicate accepting position [m] and
                velocity [m/s], returning True for an acceptable state.
            indices: Particle indices, or None for all particles. Must be nonempty.
            start_time: First required simulation time [s]. An unobserved check
                fails final validation.
            max_extent: Strict upper limit [m] on the diagonal length of the
                selected particles' axis-aligned bounding box at each observation.
                None disables this reduction. Must otherwise be finite and positive.
        """
        self._add_check(name, test_fn, "particle", indices, start_time, max_extent)

    def _add_check(self, name, test_fn, kind, indices, start_time, max_extent):
        if self._results is not None:
            raise RuntimeError("Check registration must finish before observation")
        if not name or any(check.name == name for check in self._checks):
            raise ValueError("Check names must be nonempty and unique")
        if not math.isfinite(start_time) or start_time < 0.0:
            raise ValueError("start_time must be finite and nonnegative")
        if max_extent is not None and (not math.isfinite(max_extent) or max_extent <= 0.0):
            raise ValueError("max_extent must be finite and positive")
        count = self._model.body_count if kind == "body" else self._model.particle_count
        selected = np.arange(count, dtype=np.int32) if indices is None else np.asarray(indices)
        if (
            selected.ndim != 1
            or selected.size == 0
            or not np.issubdtype(selected.dtype, np.integer)
            or np.any(selected < 0)
            or np.any(selected >= count)
        ):
            raise ValueError(f"Expected a nonempty sequence of valid {kind} indices")
        if not isinstance(test_fn, wp.Function):
            test_fn, _ = wp.utils.create_warp_function(test_fn)
        kernel = _body_kernel(test_fn) if kind == "body" else _particle_kernel(test_fn)
        self._checks.append(
            _Check(name, kind, kernel, wp.array(selected, dtype=int, device=self._model.device), start_time, max_extent)
        )

    def observe(self, state: newton.State, sim_time: float) -> None:
        """Evaluate active checks without synchronizing device state to the host.

        The first observation freezes registration and allocates reusable work
        buffers. Later observations allocate no buffers. Simulation times must
        be finite and nonnegative; invoke once after each tested step.

        Args:
            state: Body and/or particle state belonging to the registered model.
            sim_time: Simulation time [s] of this observation.
        """
        if not math.isfinite(sim_time) or sim_time < 0.0:
            raise ValueError("sim_time must be finite and nonnegative")
        if not self._checks:
            raise ValueError("Register at least one check before observation")
        if self._results is None:
            count = len(self._checks)
            device = self._model.device
            self._results = wp.zeros(count, dtype=_Result, device=device)
            self._failures = wp.empty(count, dtype=int, device=device)
            self._bounds = wp.empty((count, 6), dtype=float, device=device)
            self._start_times = wp.array([c.start_time for c in self._checks], dtype=wp.float64, device=device)
            self._extent_limits = wp.array([c.max_extent or 0.0 for c in self._checks], dtype=float, device=device)

        self._step += 1
        if all(sim_time < check.start_time for check in self._checks):
            return
        device = self._model.device
        wp.launch(_reset, dim=len(self._checks), inputs=[self._failures, self._bounds], device=device)
        for check_id, check in enumerate(self._checks):
            if sim_time < check.start_time:
                continue
            q = getattr(state, f"{check.kind}_q")
            qd = getattr(state, f"{check.kind}_qd")
            count = self._model.body_count if check.kind == "body" else self._model.particle_count
            if (
                q is None
                or qd is None
                or q.device != device
                or qd.device != device
                or q.size != count
                or qd.size != count
            ):
                raise ValueError(f"{check.name}: expected {check.kind} state on {device}")
            inputs = [q, qd, check.indices, check_id]
            if check.kind == "particle":
                inputs.append(check.max_extent is not None)
            inputs.append(self._failures)
            if check.kind == "particle":
                inputs.append(self._bounds)
            wp.launch(check.kernel, dim=check.indices.size, inputs=inputs, device=device)
        wp.launch(
            _finish,
            dim=len(self._checks),
            inputs=[
                self._failures,
                self._bounds,
                self._start_times,
                self._extent_limits,
                self._step,
                sim_time,
                self._results,
            ],
            device=device,
        )

    def assert_valid(self) -> None:
        """Read accumulated results once and raise ValueError for failed checks.

        Each registered check must have at least one eligible observation.
        Failure messages identify the first failing observation, the selected
        entity (or group extent), and its simulation time [s]. Results are not
        cleared, so repeated validation reports the same failures.
        """
        if not self._checks:
            raise ValueError("No checks registered")
        if self._results is None:
            raise ValueError(f"{', '.join(c.name for c in self._checks)}: not observed")
        errors = []
        for check, result in zip(self._checks, self._results.numpy(), strict=True):
            if result["observed"] == 0:
                errors.append(f"{check.name}: not observed (requires t >= {check.start_time:g} s)")
            elif result["step"]:
                index = int(result["index"])
                if index < 0:
                    entity = f"extent {result['extent']:.6g} m >= {check.max_extent:g} m"
                elif check.kind == "body":
                    entity = f"body {index} ({self._model.body_label[index]})"
                else:
                    entity = f"particle {index}"
                errors.append(f"{check.name}: {entity}, step {result['step']}, t={result['time']:.6g} s")
        if errors:
            raise ValueError("Example checks failed: " + "; ".join(errors))
