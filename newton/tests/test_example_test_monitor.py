# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise temporal example assertions on real body and particle states."""

import unittest

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.tests.unittest_utils import get_test_devices


class TestExampleTestMonitor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = []
        for device in get_test_devices():
            builder = newton.ModelBuilder()
            builder.add_body(label="subject")
            builder.add_body(label="excluded")
            for x in (0.0, 0.25, 10.0):
                builder.add_particle(pos=wp.vec3(x, 0.0, 0.0), vel=wp.vec3(), mass=1.0)
            cls.models.append(builder.finalize(device=device))

    def test_body_failure_survives_recovery(self):
        """Retain the first failing body and time after its velocity recovers."""
        for model in self.models:
            with self.subTest(device=model.device):
                state = model.state()
                monitor = newton.examples.ExampleTestMonitor(model)
                monitor.add_body_check("bounded speed", lambda q, qd: wp.length(qd) < 1.0, indices=[0])
                state.body_qd.assign(np.array([[0.0] * 6, [2.0] * 6], dtype=np.float32))
                monitor.observe(state, 0.1)
                monitor.assert_valid()
                state.body_qd.assign(np.full((2, 6), 2.0, dtype=np.float32))
                monitor.observe(state, 0.2)
                state.body_qd.zero_()
                monitor.observe(state, 0.3)
                with self.assertRaisesRegex(ValueError, r"bounded speed.*subject.*step 2.*0.2"):
                    monitor.assert_valid()

    def test_required_time_window(self):
        """Reject unobserved settling windows and ignore earlier motion."""
        for model in self.models:
            with self.subTest(device=model.device):
                state = model.state()
                monitor = newton.examples.ExampleTestMonitor(model)
                monitor.add_body_check("settled", lambda q, qd: wp.length(qd) < 1.0, start_time=8.0)
                monitor.add_particle_check("particle speeds", lambda q, qd: wp.length(qd) < 40.0)
                state.body_qd.fill_(wp.spatial_vector(2.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                monitor.observe(state, 7.0)
                with self.assertRaisesRegex(ValueError, "settled.*not observed"):
                    monitor.assert_valid()
                state.body_qd.zero_()
                monitor.observe(state, 8.0)
                monitor.assert_valid()

    def test_particle_predicate_and_temporal_extent(self):
        """Preserve subset finiteness and extent failures after particles recover."""
        for model in self.models:
            with self.subTest(device=model.device):
                state = model.state()
                monitor = newton.examples.ExampleTestMonitor(model)
                monitor.add_particle_check(
                    "bounded particles",
                    lambda q, qd: wp.isfinite(q[0]) and wp.length(qd) < 40.0 and q[2] > -0.2,
                    indices=[0, 1],
                    max_extent=1.0,
                )
                monitor.observe(state, 0.1)
                monitor.assert_valid()
                state.particle_q.assign(np.array([[0, 0, 0], [2, 0, 0], [10, 0, 0]], dtype=np.float32))
                monitor.observe(state, 0.2)
                state.particle_q.assign(model.particle_q)
                monitor.observe(state, 0.3)
                with self.assertRaisesRegex(ValueError, "bounded particles.*extent.*step 2"):
                    monitor.assert_valid()

                for position, velocity in (
                    ([np.nan, 0, 0], [0, 0, 0]),
                    ([0, 0, -0.2], [0, 0, 0]),
                    ([0, 0, 0], [40, 0, 0]),
                ):
                    with self.subTest(position=position, velocity=velocity):
                        monitor = newton.examples.ExampleTestMonitor(model)
                        monitor.add_particle_check(
                            "finite and bounded",
                            lambda q, qd: wp.isfinite(q[0]) and wp.length(qd) < 40.0 and q[2] > -0.2,
                            indices=[1],
                        )
                        state.particle_q.assign(np.array([[0, 0, 0], position, [10, 0, 0]], dtype=np.float32))
                        state.particle_qd.assign(np.array([[0, 0, 0], velocity, [0, 0, 0]], dtype=np.float32))
                        monitor.observe(state, 0.4)
                        state.particle_q.assign(model.particle_q)
                        state.particle_qd.zero_()
                        monitor.observe(state, 0.5)
                        with self.assertRaisesRegex(ValueError, "finite and bounded.*particle 1"):
                            monitor.assert_valid()

    def test_check_registration_contract(self):
        """Reject empty subsets, invalid indices, and registration after observation."""
        model = self.models[0]
        monitor = newton.examples.ExampleTestMonitor(model)

        def predicate(q: wp.transform, qd: wp.spatial_vector):
            return wp.length(qd) < 1.0

        for indices in ([], [-1], [model.body_count]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                monitor.add_body_check("invalid", predicate, indices=indices)
        monitor.add_body_check("valid", predicate)
        with self.assertRaisesRegex(ValueError, "not observed"):
            monitor.assert_valid()
        monitor.observe(model.state(), 0.0)
        monitor.assert_valid()
        with self.assertRaisesRegex(RuntimeError, "registration"):
            monitor.add_body_check("late", predicate)


if __name__ == "__main__":
    unittest.main()
