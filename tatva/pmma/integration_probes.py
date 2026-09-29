"""Bounded-memory, every-step shear diagnostics at selected interface nodes."""

import numpy as np


PROBE_COLUMNS = (
    "slip_rate", "cumulative_slip", "plastic_slip", "friction_coefficient",
    "signed_friction_traction", "normal_overlap", "normal_traction",
    "constrained_relative_velocity",
)


class IntegrationProbeWriter:
    """Rows i correspond to shear step i+1, not a downsampled output frame."""

    def __init__(self, h5, y, steps, dt, pressure_steps, history_columns,
                 compression, resume_step=None, buffer_steps=4096):
        self.buffer_steps = min(buffer_steps, steps)
        self.start = 0 if resume_step is None else resume_step
        self.count = 0
        n = len(y)
        self.values = np.empty((self.buffer_steps, n, len(PROBE_COLUMNS)), np.float32)
        self.state = np.empty((self.buffer_steps, n), np.float64)
        self.history = np.empty((self.buffer_steps, len(history_columns)), np.float32)
        if resume_step is None:
            g = h5.create_group("integration_probes")
            g["y"] = y
            g["columns"] = np.asarray(PROBE_COLUMNS, dtype="S")
            g["history_columns"] = np.asarray(history_columns, dtype="S")
            g.attrs.update(dt=float(dt), normal_steps=int(pressure_steps), saved_steps=0,
                           row_time="shear_time=(row_index+1)*dt; absolute_time=(normal_steps+row_index+1)*dt",
                           normal_overlap_sign="positive is compression; negative is open gap",
                           velocity_time="post-projection half-step velocity; displacement and theta at full step")
            for name, shape, dtype in (
                ("values", (steps, n, len(PROBE_COLUMNS)), "f4"),
                ("state", (steps, n), "f8"),
                ("history", (steps, len(history_columns)), "f4"),
            ):
                g.create_dataset(name, shape=shape, dtype=dtype,
                                 chunks=(self.buffer_steps,) + shape[1:],
                                 compression=compression, shuffle=True)
        self.group = h5["integration_probes"]
        if self.group["values"].shape != (steps, n, len(PROBE_COLUMNS)):
            raise ValueError("Integration probe shape changed across checkpoint resume.")
        np.testing.assert_array_equal(self.group["y"], y)
        if self.group.attrs["dt"] != float(dt):
            raise ValueError("Integration probe dt changed across checkpoint resume.")
        if resume_step is not None and int(self.group.attrs["saved_steps"]) < resume_step:
            raise ValueError("Checkpoint refers to missing integration probe rows.")
        self.group.attrs["saved_steps"] = self.start

    def append(self, start, history, values, state):
        if start != self.start + self.count:
            raise ValueError("Noncontiguous integration probe steps.")
        offset = 0
        while offset < len(history):
            count = min(len(history) - offset, self.buffer_steps - self.count)
            dest = slice(self.count, self.count + count)
            source = slice(offset, offset + count)
            self.values[dest] = values[source]
            self.state[dest] = state[source]
            self.history[dest] = history[source]
            self.count += count
            offset += count
            if self.count == self.buffer_steps:
                self.flush()

    def flush(self):
        if not self.count:
            return
        dest = slice(self.start, self.start + self.count)
        for name in ("values", "state", "history"):
            self.group[name][dest] = getattr(self, name)[:self.count]
        self.start += self.count
        self.count = 0
        self.group.attrs["saved_steps"] = self.start
