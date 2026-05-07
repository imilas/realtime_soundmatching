"""
Qt GUI for realtime hill climber optimization.

Run with:
    python gui/qt_app.py
    python -m gui.qt_app
    python -c "from gui.qt_app import main; main()"
"""

from __future__ import annotations

import random
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from threading import Event

# Ensure repo root is in path so imports work from any directory
_repo_root = Path(__file__).parent.parent.resolve()
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from agents.hillclimber import HillClimberAgent
from agents.random_search import RandomSearchAgent
from agents.q_learning import QLearningAgent
from experiments.runner import (
    ExperimentConfig,
    ExperimentRunner,
    OptimizationSnapshot,
    TARGET_RENDERED,
    TARGET_WAV,
)
from synths import SynthProgram, get_program, list_programs
from utils.loss_functions import ALL_LOSSES


# Default visualization step size
DEFAULT_STEP_PERCENT = 5.0
DEFAULT_LANDSCAPE_STEPS = 80
# Minimum |init - target| as a fraction of the param range, when both are random
MIN_DISTANCE_FRAC = 0.10
# Cap on rejection-resampling attempts when satisfying MIN_DISTANCE_FRAC
MAX_RESAMPLE_ATTEMPTS = 100


@dataclass
class ResolvedParam:
    """Concrete (init, target) pair for one parameter after resolving the spec."""
    name: str
    frozen: bool
    init: float
    target: float


class ParamRow(QtWidgets.QGroupBox):
    """Per-parameter configuration: freeze + per-field override (init/target/frozen).

    Two orthogonal axes:
      Frozen / not frozen — does the agent optimize this param?
      Override / random   — is each value user-specified or drawn at random?
    """

    def __init__(self, name: str, min_val: float, max_val: float, step: float, parent=None):
        super().__init__(name, parent)
        self.name = name
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.step = float(step) if step and step > 0 else 0.0

        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setVerticalSpacing(4)

        step_str = f", step {self.step:g}" if self.step > 0 else ""
        range_label = QtWidgets.QLabel(f"Range [{self.min_val:g}, {self.max_val:g}]{step_str}")
        range_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(range_label, 0, 0, 1, 4)

        self.freeze_check = QtWidgets.QCheckBox("Freeze (exclude from optimization)")
        self.freeze_check.toggled.connect(self._on_freeze_toggled)
        layout.addWidget(self.freeze_check, 1, 0, 1, 4)

        self._init_widgets = self._build_value_row(layout, 2, "Init", "#1f77b4")
        self._target_widgets = self._build_value_row(layout, 3, "Target", "#d62728")
        self._frozen_widgets = self._build_value_row(layout, 4, "Frozen value", "#555555")

        # Live "current" value updated by agent snapshots
        cur_label = QtWidgets.QLabel("Current:")
        cur_label.setStyleSheet("color: gray; font-size: 10px;")
        self._current_display = QtWidgets.QLabel("—")
        self._current_display.setStyleSheet("color: #2ca02c; font-weight: bold;")
        layout.addWidget(cur_label, 5, 0)
        layout.addWidget(self._current_display, 5, 1, 1, 3)

        self._on_freeze_toggled(False)

    def _build_value_row(self, layout: QtWidgets.QGridLayout, row_idx: int, label_text: str, color: str):
        label = QtWidgets.QLabel(f"{label_text}:")
        override = QtWidgets.QCheckBox("Override")
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(self.min_val, self.max_val)
        spin.setSingleStep(self.step if self.step > 0 else max((self.max_val - self.min_val) / 100, 0.001))
        spin.setDecimals(0 if self.step >= 1 else 4)
        spin.setValue((self.min_val + self.max_val) / 2)
        spin.setEnabled(False)
        override.toggled.connect(spin.setEnabled)
        resolved = QtWidgets.QLabel("—")
        resolved.setStyleSheet(f"color: {color}; font-weight: bold; min-width: 80px;")
        layout.addWidget(label, row_idx, 0)
        layout.addWidget(override, row_idx, 1)
        layout.addWidget(spin, row_idx, 2)
        layout.addWidget(resolved, row_idx, 3)
        return {"label": label, "override": override, "spin": spin, "resolved": resolved}

    def _on_freeze_toggled(self, frozen: bool):
        for w in self._init_widgets.values():
            w.setVisible(not frozen)
        for w in self._target_widgets.values():
            w.setVisible(not frozen)
        for w in self._frozen_widgets.values():
            w.setVisible(frozen)

    def is_frozen(self) -> bool:
        return self.freeze_check.isChecked()

    def quantize(self, v: float) -> float:
        v = max(self.min_val, min(self.max_val, v))
        if self.step > 0:
            v = round((v - self.min_val) / self.step) * self.step + self.min_val
            v = max(self.min_val, min(self.max_val, v))
        return v

    def _format_value(self, v: float) -> str:
        if self.step >= 1:
            return f"{int(round(v))}"
        return f"{v:.4f}"

    def resolve(self, rng: random.Random, min_distance_frac: float) -> tuple[ResolvedParam, int, bool]:
        """Compute concrete (init, target). Returns (resolved, attempts, distance_violated)."""
        wi, wt, wf = self._init_widgets, self._target_widgets, self._frozen_widgets

        if self.is_frozen():
            if wf["override"].isChecked():
                v = self.quantize(wf["spin"].value())
            else:
                v = self.quantize(rng.uniform(self.min_val, self.max_val))
            wf["resolved"].setText(self._format_value(v))
            wi["resolved"].setText("—")
            wt["resolved"].setText("—")
            return ResolvedParam(self.name, frozen=True, init=v, target=v), 1, False

        init_override = wi["override"].isChecked()
        target_override = wt["override"].isChecked()
        range_size = self.max_val - self.min_val
        min_dist = min_distance_frac * range_size

        if init_override and target_override:
            i = self.quantize(wi["spin"].value())
            t = self.quantize(wt["spin"].value())
            attempts = 1
            violated = abs(i - t) < min_dist
        else:
            attempts = 0
            i = t = 0.0
            for k in range(MAX_RESAMPLE_ATTEMPTS):
                attempts = k + 1
                i = self.quantize(wi["spin"].value()) if init_override else self.quantize(rng.uniform(self.min_val, self.max_val))
                t = self.quantize(wt["spin"].value()) if target_override else self.quantize(rng.uniform(self.min_val, self.max_val))
                if abs(i - t) >= min_dist:
                    break
            violated = abs(i - t) < min_dist

        wi["resolved"].setText(self._format_value(i))
        wt["resolved"].setText(self._format_value(t))
        wf["resolved"].setText("—")
        return ResolvedParam(self.name, frozen=False, init=i, target=t), attempts, violated

    def set_current_value(self, value: float):
        self._current_display.setText(self._format_value(value))

    def clear_current_value(self):
        self._current_display.setText("—")


class LandscapeWorkerThread(QtCore.QObject):
    """Worker thread for computing loss landscape."""
    point_ready = QtCore.Signal(int, float, float)
    finished = QtCore.Signal(np.ndarray, np.ndarray)
    failed = QtCore.Signal(str)

    def __init__(self, runner: ExperimentRunner):
        super().__init__()
        self.runner = runner
        self._cancel = Event()

    @QtCore.Slot()
    def run(self):
        try:
            values = []
            losses = []
            for point in self.runner.compute_landscape():
                if self._cancel.is_set():
                    return
                values.append(point.value)
                losses.append(point.loss)
                self.point_ready.emit(point.index, point.value, point.loss)
            self.finished.emit(np.array(values), np.array(losses))
        except Exception:
            self.failed.emit(traceback.format_exc())

    def cancel(self):
        self._cancel.set()


class OptimizationWorkerThread(QtCore.QObject):
    """Worker thread for running optimization."""
    snapshot_ready = QtCore.Signal(object)
    status = QtCore.Signal(str)
    finished = QtCore.Signal()
    failed = QtCore.Signal(str)

    def __init__(self, runner: ExperimentRunner, agent_type: str, step_percent: float):
        super().__init__()
        self.runner = runner
        self.agent_type = agent_type
        self.step_percent = step_percent
        self._stop = Event()

    @QtCore.Slot()
    def run(self):
        try:
            if self.agent_type == "hillclimber":
                agent = HillClimberAgent(step_percent=self.step_percent)
            elif self.agent_type == "random":
                agent = RandomSearchAgent(step_percent=self.step_percent)
            elif self.agent_type == "q_learning":
                agent = QLearningAgent(step_percent=self.step_percent)
            else:
                raise ValueError(f"Unknown agent type: {self.agent_type}")

            self.status.emit("Agent running")
            for snapshot in self.runner.run_optimization(
                agent, stop_check=lambda: self._stop.is_set()
            ):
                self.snapshot_ready.emit(snapshot)

            self.status.emit("Agent stopped")
            self.finished.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())

    def stop(self):
        self._stop.set()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Realtime Optimizer")
        self.resize(1600, 950)

        self.repo_root = Path(__file__).parent.parent.resolve()
        self.program: SynthProgram | None = None
        self.param_rows: dict[str, ParamRow] = {}
        self.last_resolved: dict[str, ResolvedParam] = {}
        self.agent_thread: QtCore.QThread | None = None
        self.agent_worker: OptimizationWorkerThread | None = None
        self.landscape_thread: QtCore.QThread | None = None
        self.landscape_worker: LandscapeWorkerThread | None = None
        self.landscape_x = np.array([], dtype=np.float64)
        self.landscape_y = np.array([], dtype=np.float64)
        self.latest_snapshot: OptimizationSnapshot | None = None

        self._build_ui()
        self._configure_plots()
        self.load_program()

    def _build_ui(self):
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QHBoxLayout(root)

        controls = QtWidgets.QWidget()
        controls.setMaximumWidth(460)
        controls_layout = QtWidgets.QVBoxLayout(controls)

        # Program selection
        synth_group = QtWidgets.QGroupBox("Synth Program")
        synth_layout = QtWidgets.QVBoxLayout(synth_group)
        self.synth_combo = QtWidgets.QComboBox()
        programs = list_programs()
        self.synth_combo.addItems(programs)
        if "bandpass_noise" in programs:
            self.synth_combo.setCurrentText("bandpass_noise")
        self.synth_combo.currentTextChanged.connect(self._on_synth_changed)
        synth_layout.addWidget(QtWidgets.QLabel("Available programs:"))
        synth_layout.addWidget(self.synth_combo)
        controls_layout.addWidget(synth_group)

        # Configuration (sweep param is auto-derived from the single non-frozen param)
        config_group = QtWidgets.QGroupBox("Configuration")
        config_form = QtWidgets.QFormLayout(config_group)
        self.agent_combo = QtWidgets.QComboBox()
        self.agent_combo.addItems(["hillclimber", "random", "q_learning"])
        self.loss_combo = QtWidgets.QComboBox()
        self.loss_combo.addItems(list(ALL_LOSSES.keys()))
        self.loss_combo.setCurrentText("Multi-Res Spectral")
        self.step_percent_input = QtWidgets.QDoubleSpinBox()
        self.step_percent_input.setRange(0.1, 100.0)
        self.step_percent_input.setDecimals(2)
        self.step_percent_input.setValue(DEFAULT_STEP_PERCENT)
        self.landscape_steps_input = QtWidgets.QSpinBox()
        self.landscape_steps_input.setRange(8, 400)
        self.landscape_steps_input.setValue(DEFAULT_LANDSCAPE_STEPS)
        config_form.addRow("Agent Type", self.agent_combo)
        config_form.addRow("Loss Function", self.loss_combo)
        config_form.addRow("Step Size (%)", self.step_percent_input)
        config_form.addRow("Landscape Steps", self.landscape_steps_input)
        controls_layout.addWidget(config_group)

        # Target source — rendered (in-domain) or WAV file (out-of-domain)
        target_group = QtWidgets.QGroupBox("Target Source")
        target_layout = QtWidgets.QVBoxLayout(target_group)
        self.target_rendered_radio = QtWidgets.QRadioButton("Rendered (in-domain)")
        self.target_wav_radio = QtWidgets.QRadioButton("WAV file (out-of-domain)")
        self.target_rendered_radio.setChecked(True)
        target_layout.addWidget(self.target_rendered_radio)
        target_layout.addWidget(self.target_wav_radio)
        wav_row = QtWidgets.QHBoxLayout()
        self.target_wav_path_edit = QtWidgets.QLineEdit()
        self.target_wav_path_edit.setPlaceholderText("path to .wav")
        self.target_wav_path_edit.setEnabled(False)
        self.target_wav_browse = QtWidgets.QPushButton("Browse…")
        self.target_wav_browse.setEnabled(False)
        wav_row.addWidget(self.target_wav_path_edit)
        wav_row.addWidget(self.target_wav_browse)
        target_layout.addLayout(wav_row)

        def _on_wav_toggled(checked: bool):
            self.target_wav_path_edit.setEnabled(checked)
            self.target_wav_browse.setEnabled(checked)
        self.target_wav_radio.toggled.connect(_on_wav_toggled)
        self.target_wav_browse.clicked.connect(self._browse_target_wav)
        controls_layout.addWidget(target_group)

        # Parameters
        params_group = QtWidgets.QGroupBox("Parameter Configuration")
        params_outer = QtWidgets.QVBoxLayout(params_group)

        seed_row = QtWidgets.QHBoxLayout()
        seed_row.addWidget(QtWidgets.QLabel("Seed:"))
        self.seed_input = QtWidgets.QLineEdit()
        self.seed_input.setPlaceholderText("(empty = nondeterministic)")
        self.seed_input.setMaximumWidth(140)
        seed_row.addWidget(self.seed_input)
        self.resolve_button = QtWidgets.QPushButton("Resolve / Randomize")
        seed_row.addWidget(self.resolve_button)
        seed_row.addStretch(1)
        params_outer.addLayout(seed_row)

        self.params_container = QtWidgets.QWidget()
        self.params_container_layout = QtWidgets.QVBoxLayout(self.params_container)
        self.params_container_layout.setContentsMargins(0, 0, 0, 0)
        self.params_container_layout.setSpacing(6)
        params_outer.addWidget(self.params_container)

        self.resolved_display = QtWidgets.QPlainTextEdit()
        self.resolved_display.setReadOnly(True)
        self.resolved_display.setMaximumHeight(120)
        self.resolved_display.setPlaceholderText("Click Resolve / Randomize to see init/target values.")
        params_outer.addWidget(self.resolved_display)

        controls_layout.addWidget(params_group)

        # Actions
        action_row = QtWidgets.QHBoxLayout()
        self.update_landscape_button = QtWidgets.QPushButton("Update Landscape")
        self.start_button = QtWidgets.QPushButton("Start Agent")
        self.stop_button = QtWidgets.QPushButton("Stop Agent")
        self.stop_button.setEnabled(False)
        action_row.addWidget(self.update_landscape_button)
        action_row.addWidget(self.start_button)
        action_row.addWidget(self.stop_button)
        controls_layout.addLayout(action_row)

        # Metrics
        metrics_group = QtWidgets.QGroupBox("Live Metrics")
        metrics_form = QtWidgets.QFormLayout(metrics_group)
        self.current_loss_label = QtWidgets.QLabel("—")
        self.best_loss_label = QtWidgets.QLabel("—")
        self.current_value_label = QtWidgets.QLabel("—")
        self.best_value_label = QtWidgets.QLabel("—")
        self.iteration_label = QtWidgets.QLabel("0")
        metrics_form.addRow("Current Loss", self.current_loss_label)
        metrics_form.addRow("Best Loss", self.best_loss_label)
        metrics_form.addRow("Current Value", self.current_value_label)
        metrics_form.addRow("Best Value", self.best_value_label)
        metrics_form.addRow("Iterations", self.iteration_label)
        controls_layout.addWidget(metrics_group)

        self.status_label = QtWidgets.QLabel("Idle")
        self.status_label.setWordWrap(True)
        controls_layout.addWidget(self.status_label)
        controls_layout.addStretch(1)

        # Plots
        plots = QtWidgets.QWidget()
        plots_layout = QtWidgets.QVBoxLayout(plots)
        self.graphics = pg.GraphicsLayoutWidget()
        plots_layout.addWidget(self.graphics)

        layout.addWidget(controls)
        layout.addWidget(plots, stretch=1)

        # Connect signals
        self.update_landscape_button.clicked.connect(self.recompute_landscape)
        self.start_button.clicked.connect(self.start_agent)
        self.stop_button.clicked.connect(self.stop_agent)
        self.resolve_button.clicked.connect(self.resolve_params)

    def _configure_plots(self):
        pg.setConfigOptions(antialias=True)
        self.landscape_plot = self.graphics.addPlot(row=0, col=0, title="Loss Landscape")
        self.landscape_plot.showGrid(x=True, y=True, alpha=0.25)
        self.landscape_plot.setLabel("left", "Loss")
        self.landscape_plot.setLabel("bottom", "Parameter Value")
        self.landscape_curve = self.landscape_plot.plot(pen=pg.mkPen("#1f77b4", width=2))
        self.trail_scatter = pg.ScatterPlotItem(size=8, brush=pg.mkBrush(255, 153, 0, 120), pen=None)
        self.best_scatter = pg.ScatterPlotItem(
            size=14, brush=pg.mkBrush("#2ca02c"), pen=pg.mkPen("#1b5e20", width=1.5), symbol="s"
        )
        self.current_scatter = pg.ScatterPlotItem(
            size=18, brush=pg.mkBrush("#d62728"), pen=pg.mkPen("#7f0000", width=1.5), symbol="star"
        )
        self.landscape_plot.addItem(self.trail_scatter)
        self.landscape_plot.addItem(self.best_scatter)
        self.landscape_plot.addItem(self.current_scatter)

        self.loss_plot = self.graphics.addPlot(row=1, col=0, title="Live Loss")
        self.loss_plot.showGrid(x=True, y=True, alpha=0.25)
        self.loss_plot.setLabel("left", "Loss")
        self.loss_plot.setLabel("bottom", "Iteration")
        self.loss_curve = self.loss_plot.plot(pen=pg.mkPen("#444444", width=2))
        self.best_loss_curve = self.loss_plot.plot(
            pen=pg.mkPen("#2ca02c", width=1.5, style=QtCore.Qt.DashLine)
        )

    def _set_status(self, text: str):
        self.status_label.setText(text)

    def _on_synth_changed(self, synth_name: str):
        """Load a different program."""
        self.load_program(synth_name)

    def load_program(self, name: str | None = None):
        """Load a SynthProgram (parses bounds from the .dsp template). No compile yet."""
        if name is None:
            name = self.synth_combo.currentText() if hasattr(self, "synth_combo") else "bandpass_noise"

        try:
            self.program = get_program(name)

            while self.params_container_layout.count():
                item = self.params_container_layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.deleteLater()
            self.param_rows.clear()
            self.last_resolved.clear()
            self.resolved_display.setPlainText("")

            for pname, (lo, hi) in self.program.param_ranges.items():
                # Parse step from the template if present (hslider's 5th arg)
                step = self._parse_step(pname)
                row = ParamRow(pname, lo, hi, step)
                self.params_container_layout.addWidget(row)
                self.param_rows[pname] = row

            self._set_status(f"Loaded program '{name}' — set freeze/overrides, then Resolve")
        except Exception as exc:
            self._set_status(f"Failed to load program '{name}': {exc}")

    def _parse_step(self, param_name: str) -> float:
        """Pull step from the hslider definition for `param_name`. Returns 0.0 if absent."""
        import re
        if self.program is None:
            return 0.0
        pattern = (
            r'hslider\s*\(\s*"' + re.escape(param_name)
            + r'"\s*,\s*\{[^}]*\}\s*,\s*[^,]+\s*,\s*[^,]+\s*,\s*([^\)]+)\)'
        )
        m = re.search(pattern, self.program.faust_template)
        if not m:
            return 0.0
        try:
            return float(m.group(1).strip())
        except ValueError:
            return 0.0

    def _browse_target_wav(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select target WAV", str(self.repo_root), "WAV files (*.wav)"
        )
        if path:
            self.target_wav_path_edit.setText(path)
            self.target_wav_radio.setChecked(True)

    def _build_experiment_config(self) -> ExperimentConfig | None:
        """Construct ExperimentConfig from the current resolved params + target source.

        Returns None and updates the status bar if the GUI is not in a runnable state.
        """
        if self.program is None or not self.last_resolved:
            self._set_status("Click Resolve / Randomize before running")
            return None

        init = {n: rp.init for n, rp in self.last_resolved.items()}
        target = {n: rp.target for n, rp in self.last_resolved.items()}
        frozen = {n for n, rp in self.last_resolved.items() if rp.frozen}

        if self.target_wav_radio.isChecked():
            wav_path = self.target_wav_path_edit.text().strip()
            if not wav_path:
                self._set_status("Pick a WAV file or switch to Rendered target")
                return None
            source = TARGET_WAV
        else:
            wav_path = None
            source = TARGET_RENDERED

        try:
            return ExperimentConfig(
                program_name=self.program.name,
                init_params=init,
                target_params=target,
                frozen_params=frozen,
                target_source=source,
                target_wav_path=wav_path,
                loss_name=self.loss_combo.currentText(),
                landscape_steps=self.landscape_steps_input.value(),
            )
        except ValueError as exc:
            self._set_status(f"Invalid configuration: {exc}")
            return None

    def current_param_values(self) -> dict[str, float]:
        """Return init values from the last Resolve, falling back to mid-of-range."""
        if self.last_resolved:
            return {name: rp.init for name, rp in self.last_resolved.items()}
        return {
            name: (row.min_val + row.max_val) / 2
            for name, row in self.param_rows.items()
        }

    def _sweep_param(self) -> str | None:
        """The single non-frozen param, if exactly one. Else None."""
        if not self.last_resolved:
            return None
        non_frozen = [n for n, rp in self.last_resolved.items() if not rp.frozen]
        return non_frozen[0] if len(non_frozen) == 1 else None

    def resolve_params(self):
        if not self.param_rows:
            self._set_status("No synth loaded")
            return

        seed_text = self.seed_input.text().strip()
        try:
            seed = int(seed_text) if seed_text else None
        except ValueError:
            self._set_status(f"Invalid seed: {seed_text!r} (must be an integer)")
            return
        rng = random.Random(seed)

        resolved: list[ResolvedParam] = []
        max_attempts = 0
        violations: list[str] = []
        for name, row in self.param_rows.items():
            rp, attempts, violated = row.resolve(rng, MIN_DISTANCE_FRAC)
            resolved.append(rp)
            max_attempts = max(max_attempts, attempts)
            if violated:
                violations.append(name)

        self.last_resolved = {rp.name: rp for rp in resolved}

        n_optim = sum(1 for rp in resolved if not rp.frozen)
        n_frozen = sum(1 for rp in resolved if rp.frozen)
        lines = [
            f"Optimized: {n_optim}   Frozen: {n_frozen}",
            f"Seed: {seed if seed is not None else '(nondeterministic)'}   "
            f"Max resampling attempts: {max_attempts}",
        ]
        if violations:
            lines.append(
                f"WARNING: 10% min-distance violated by user override on: {', '.join(violations)}"
            )
        lines.append("")
        for rp in resolved:
            row = self.param_rows[rp.name]
            if rp.frozen:
                lines.append(f"  {rp.name}: FROZEN at {row._format_value(rp.init)}")
            else:
                lines.append(
                    f"  {rp.name}: init={row._format_value(rp.init)}  "
                    f"target={row._format_value(rp.target)}  "
                    f"|Δ|={abs(rp.init - rp.target):g}"
                )
        if n_optim == 0:
            lines.append("")
            lines.append("ERROR: every parameter is frozen — nothing to optimize.")

        self.resolved_display.setPlainText("\n".join(lines))
        self._set_status(f"Resolved {len(resolved)} param(s)")

    def _refresh_markers_only(self):
        sweep_param = self._sweep_param()
        if sweep_param is None:
            return
        current_value = self.current_param_values().get(sweep_param)
        if current_value is None:
            return
        if self.landscape_x.size and self.landscape_y.size:
            current_loss = float(np.interp(current_value, self.landscape_x, self.landscape_y))
            self.current_scatter.setData([current_value], [current_loss])
            self.trail_scatter.setData([], [])
            self.best_scatter.setData([], [])

    def recompute_landscape(self):
        if self.agent_worker is not None or self.landscape_thread is not None:
            self._set_status("Busy — finish current operation first")
            return

        config = self._build_experiment_config()
        if config is None:
            return
        try:
            runner = ExperimentRunner(config, self.repo_root)
        except Exception as exc:
            self._set_status(f"Cannot build runner: {exc}")
            return

        sweep_param = runner.sweep_param
        target_value = config.target_params[sweep_param]

        self.landscape_x = np.array([], dtype=np.float64)
        self.landscape_y = np.full_like(self.landscape_x, np.nan, dtype=np.float64)
        self.landscape_curve.setData(self.landscape_x, self.landscape_y)
        self.trail_scatter.setData([], [])
        self.best_scatter.setData([], [])
        self.current_scatter.setData([], [])

        self.landscape_thread = QtCore.QThread(self)
        self.landscape_worker = LandscapeWorkerThread(runner)
        self.landscape_worker.moveToThread(self.landscape_thread)
        self.landscape_thread.started.connect(self.landscape_worker.run)
        self.landscape_worker.point_ready.connect(self._on_landscape_point)
        self.landscape_worker.finished.connect(self._on_landscape_finished)
        self.landscape_worker.failed.connect(self._on_landscape_failed)
        self.landscape_worker.finished.connect(self.landscape_thread.quit)
        self.landscape_worker.failed.connect(self.landscape_thread.quit)
        self.landscape_thread.finished.connect(self._cleanup_landscape_worker)
        self.landscape_thread.start()
        self._set_status(
            f"Computing landscape ({sweep_param}, target={target_value:g}, "
            f"source={config.target_source}, loss={config.loss_name})..."
        )

    def _cleanup_landscape_worker(self):
        if self.landscape_worker is not None:
            self.landscape_worker.deleteLater()
        if self.landscape_thread is not None:
            self.landscape_thread.deleteLater()
        self.landscape_worker = None
        self.landscape_thread = None

    def _on_landscape_point(self, idx: int, value: float, loss: float):
        if not self.landscape_x.size:
            self.landscape_x = np.array([], dtype=np.float64)
            self.landscape_y = np.array([], dtype=np.float64)
        self.landscape_x = np.append(self.landscape_x, value)
        self.landscape_y = np.append(self.landscape_y, loss)
        self.landscape_curve.setData(self.landscape_x, self.landscape_y)

    def _on_landscape_finished(self, sweep_values: np.ndarray, losses: np.ndarray):
        self.landscape_x = sweep_values
        self.landscape_y = losses
        self.landscape_curve.setData(self.landscape_x, self.landscape_y)
        self._refresh_markers_only()
        self._set_status("Landscape updated")

    def _on_landscape_failed(self, trace: str):
        self._set_status(f"Landscape update failed:\n{trace}")

    def _set_running(self, running: bool):
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.synth_combo.setEnabled(not running)
        self.update_landscape_button.setEnabled(not running)
        self.agent_combo.setEnabled(not running)
        self.loss_combo.setEnabled(not running)
        self.step_percent_input.setEnabled(not running)
        self.landscape_steps_input.setEnabled(not running)
        self.resolve_button.setEnabled(not running)
        self.seed_input.setEnabled(not running)
        self.target_rendered_radio.setEnabled(not running)
        self.target_wav_radio.setEnabled(not running)
        self.target_wav_path_edit.setEnabled(
            not running and self.target_wav_radio.isChecked()
        )
        self.target_wav_browse.setEnabled(
            not running and self.target_wav_radio.isChecked()
        )
        for row in self.param_rows.values():
            row.setEnabled(not running)

    def start_agent(self):
        if self.agent_thread is not None:
            return

        config = self._build_experiment_config()
        if config is None:
            return
        try:
            runner = ExperimentRunner(config, self.repo_root)
        except Exception as exc:
            self._set_status(f"Cannot build runner: {exc}")
            return

        self.agent_thread = QtCore.QThread(self)
        self.agent_worker = OptimizationWorkerThread(
            runner,
            self.agent_combo.currentText(),
            self.step_percent_input.value(),
        )
        self.agent_worker.moveToThread(self.agent_thread)
        self.agent_thread.started.connect(self.agent_worker.run)
        self.agent_worker.snapshot_ready.connect(self._on_agent_snapshot)
        self.agent_worker.status.connect(self._set_status)
        self.agent_worker.failed.connect(self._on_agent_failed)
        self.agent_worker.finished.connect(self.agent_thread.quit)
        self.agent_thread.finished.connect(self._on_agent_thread_finished)
        self._set_running(True)
        self.agent_thread.start()

    def stop_agent(self):
        if self.agent_worker is not None:
            self.agent_worker.stop()

    def _on_agent_snapshot(self, snapshot: OptimizationSnapshot):
        self.latest_snapshot = snapshot
        self.iteration_label.setText(str(snapshot.iteration))
        self.current_loss_label.setText(f"{snapshot.current_loss:.5f}")
        self.best_loss_label.setText(f"{snapshot.best_loss:.5f}")
        self.current_value_label.setText(f"{snapshot.current_value:.3f}")
        self.best_value_label.setText(f"{snapshot.best_value:.3f}")

        if self.landscape_x.size and self.landscape_y.size:
            trail_x = np.array(snapshot.history_values[-40:], dtype=np.float64)
            trail_y = np.interp(trail_x, self.landscape_x, self.landscape_y)
            self.trail_scatter.setData(trail_x, trail_y)
            best_y = float(np.interp(snapshot.best_value, self.landscape_x, self.landscape_y))
            current_y = float(np.interp(snapshot.current_value, self.landscape_x, self.landscape_y))
            self.best_scatter.setData([snapshot.best_value], [best_y])
            self.current_scatter.setData([snapshot.current_value], [current_y])

        xs = np.arange(len(snapshot.history_losses), dtype=np.float64)
        ys = np.array(snapshot.history_losses, dtype=np.float64)
        best_series = np.minimum.accumulate(ys)
        self.loss_curve.setData(xs, ys)
        self.best_loss_curve.setData(xs, best_series)

        for name, value in snapshot.current_params.items():
            row = self.param_rows.get(name)
            if row is not None:
                row.set_current_value(value)

    def _on_agent_failed(self, trace: str):
        self._set_status(f"Agent failed:\n{trace}")
        self.stop_agent()

    def _on_agent_thread_finished(self):
        if self.agent_worker is not None:
            self.agent_worker.deleteLater()
        if self.agent_thread is not None:
            self.agent_thread.deleteLater()
        self.agent_worker = None
        self.agent_thread = None
        self._set_running(False)

    def closeEvent(self, event):
        if self.landscape_worker is not None:
            self.landscape_worker.cancel()
        if self.agent_worker is not None:
            self.agent_worker.stop()
        if self.agent_thread is not None:
            self.agent_thread.quit()
            self.agent_thread.wait(3000)
        if self.landscape_thread is not None:
            self.landscape_thread.quit()
            self.landscape_thread.wait(3000)
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
