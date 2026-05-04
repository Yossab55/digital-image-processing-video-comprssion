import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.io import wavfile


# ══════════════════════════════════════════════════════════════════════════════
#  DSP Processing Functions  (exact logic from the original script)
# ══════════════════════════════════════════════════════════════════════════════

def load_audio(path):
    """Read WAV, convert stereo→mono, return float64 signal."""
    Fs, audio = wavfile.read(path)
    dtype = audio.dtype
    if dtype == np.int16:
        bits = 16
    elif dtype == np.int32:
        bits = 32
    elif dtype == np.uint8:
        bits = 8
    else:
        bits = 16

    # stereo → mono  (original script: mean of channels)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float64)
    return Fs, audio, bits


def add_noise_and_silence(audio, Fs):
    """Add Gaussian noise and 1-second silence pads."""
    noise           = 0.03 * np.random.randn(len(audio))
    audio_noisy     = audio + noise
    silence         = np.zeros(int(Fs * 1))
    audio_with_sil  = np.concatenate((silence, audio_noisy, silence))
    return audio_noisy, audio_with_sil


def compute_stft(signal, frame_size=256, hop=128):
    """Manual STFT with Hamming-like window."""
    N      = frame_size
    window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))
    nf     = int(np.floor((len(signal) - frame_size) / hop)) + 1
    S      = np.zeros((frame_size, nf), dtype=complex)
    for i in range(nf):
        start     = i * hop
        frame     = signal[start:start + frame_size] * window
        S[:, i]   = np.fft.fft(frame)
    return S


def quantize(S, levels=8):
    """Magnitude normalisation + uniform quantisation."""
    S_abs   = np.abs(S)
    S_max   = np.max(S_abs)
    S_norm  = S_abs / S_max
    S_quant = np.round(S_norm * (levels - 1)).astype(int)
    return S_abs, S_max, S_norm, S_quant


def rle_encode(S_quant):
    """Run-length encode the flattened quantised spectrogram."""
    data    = S_quant.flatten()
    values, counts = [], []
    current, count = data[0], 1
    for val in data[1:]:
        if val == current:
            count += 1
        else:
            values.append(current); counts.append(count)
            current, count = val, 1
    values.append(current); counts.append(count)
    return data, values, counts


def rle_decode(values, counts, shape):
    """Decode RLE back to array."""
    rec = []
    for v, c in zip(values, counts):
        rec.extend([v] * c)
    return np.array(rec).reshape(shape)


def reconstruct_signal(S_rec, S_max, ref_signal, levels=8,
                        frame_size=256, hop=128):
    """Dequantise + IFFT overlap-add reconstruction."""
    S_rec_abs  = (S_rec / (levels - 1)) * S_max
    signal_rec = np.zeros(len(ref_signal))
    for i in range(S_rec.shape[1]):
        start = i * hop
        if start + frame_size > len(signal_rec):
            break
        signal_rec[start:start + frame_size] += np.fft.ifft(S_rec_abs[:, i]).real
    return signal_rec


def compute_snr(orig, rec):
    n = min(len(orig), len(rec))
    o, r = orig[:n], rec[:n]
    return 10 * np.log10(np.sum(o2) / np.sum((o - r)2))


# ══════════════════════════════════════════════════════════════════════════════
#  GUI
# ══════════════════════════════════════════════════════════════════════════════

BG = "#d9d9d9"

PLOT_NAMES = [
    "Original Audio Signal",
    "Audio with Noise and Silence",
    "Spectrogram (Manual)",
    "Before Quantization",
    "After Quantization",
    "Original vs Reconstructed",
]


class DSPProjectGUI:
    def init(self, root):
        self.root = root
        self.root.title("DSP_Project")
        self.root.configure(bg=BG)
        self.root.minsize(860, 620)\
  
# ── application state ──────────────────────────────────────────────
        self.Fs                 = None
        self.audio              = None
        self.bits               = None
        self.audio_with_silence = None
        self.signal_rec         = None
        self.plot_data          = {}   # filled after pipeline run

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # top controls
        ctrl = tk.Frame(self.root, bg=BG)
        ctrl.pack(fill=tk.X, padx=10, pady=(10, 4))

        tk.Button(ctrl, text="Load Audio", command=self._load_audio,
                  relief=tk.GROOVE, bg=BG, padx=8, pady=2,
                  font=("TkDefaultFont", 9)).pack(side=tk.LEFT)

        self.run_btn = tk.Button(ctrl, text="Run Full Pipeline",
                                 command=self._run_pipeline,
                                 relief=tk.GROOVE, bg=BG, padx=8, pady=2,
                                 font=("TkDefaultFont", 9), state=tk.DISABLED)
        self.run_btn.pack(side=tk.LEFT, padx=(8, 0))

        tk.Label(ctrl, text="  View:", bg=BG,
                 font=("TkDefaultFont", 9)).pack(side=tk.LEFT, padx=(20, 0))

        self.plot_var = tk.StringVar(value=PLOT_NAMES[0])
        self.plot_menu = tk.OptionMenu(ctrl, self.plot_var, *PLOT_NAMES,
                                       command=self._switch_plot)
        self.plot_menu.config(bg=BG, relief=tk.GROOVE,
                              font=("TkDefaultFont", 9))
        self.plot_menu.pack(side=tk.LEFT, padx=4)

        # matplotlib canvas
        pf = tk.Frame(self.root, bg=BG)
        pf.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        self.fig = plt.Figure(figsize=(9, 4), facecolor=BG)
        self.ax  = self.fig.add_subplot(111)
        self.ax.set_facecolor("white")
        self.fig.tight_layout(pad=1.8)

        self.canvas = FigureCanvasTkAgg(self.fig, master=pf)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        tb_frame = tk.Frame(pf, bg=BG)
        tb_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(self.canvas, tb_frame)

        # status
        self.status_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.status_var, bg=BG,
                 font=("TkDefaultFont", 9)).pack(pady=(0, 2))

        # info labels
        info = tk.Frame(self.root, bg=BG)
        info.pack(fill=tk.X, padx=20, pady=(0, 8))

        self.duration_var = tk.StringVar(value="Duration:")
        self.fs_var       = tk.StringVar(value="FS:")
        self.levels_var   = tk.StringVar(value="Levels:")
        self.bps_var      = tk.StringVar(value="Bits per sample:")
        self.orig_sz_var  = tk.StringVar(value="Original Size:")
        self.comp_sz_var  = tk.StringVar(value="Compressed Size:")
        self.snr_var      = tk.StringVar(value="SNR:")

        row0 = [self.duration_var, self.fs_var, self.levels_var]
        row1 = [self.bps_var, self.orig_sz_var, self.comp_sz_var]

        for c, v in enumerate(row0):
            tk.Label(info, textvariable=v, bg=BG,
                     font=("TkDefaultFont", 9)).grid(
                row=0, column=c, sticky="w", padx=(0, 50))

        for c, v in enumerate(row1):
            tk.Label(info, textvariable=v, bg=BG,
                     font=("TkDefaultFont", 9)).grid(
                row=1, column=c, sticky="w", padx=(0, 50), pady=(4, 0))

        tk.Label(info, textvariable=self.snr_var, bg=BG,
                 font=("TkDefaultFont", 9, "bold")).grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))

    # ── Load Audio ────────────────────────────────────────────────────────────
    def _load_audio(self):
        path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if not path:
            return

try:
            self.Fs, self.audio, self.bits = load_audio(path)
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            return

        # console output matching original script
        print("Sampling Frequency:", self.Fs)
        print("Audio Length:",       len(self.audio))

        duration = len(self.audio) / self.Fs
        levels   = 2 ** self.bits

        self.duration_var.set(f"Duration: {duration:.3f} s")
        self.fs_var.set(f"FS: {self.Fs} Hz")
        self.levels_var.set(f"Levels: {levels}")
        self.bps_var.set(f"Bits per sample: {self.bits}")
        self.status_var.set("File loaded Sucessfully.")

        self.run_btn.config(state=tk.NORMAL)
        self.plot_data = {}

        # immediately show original waveform
        t = np.arange(len(self.audio)) / self.Fs
        self._draw_line(t, self.audio,
                        "Original Audio Signal", "Time", "Amplitude")
        self.plot_var.set("Original Audio Signal")

    # ── Run Full Pipeline ─────────────────────────────────────────────────────
    def _run_pipeline(self):
        if self.audio is None:
            return

        self.status_var.set("Running pipeline…")
        self.root.update_idletasks()

        Fs    = self.Fs
        audio = self.audio

        # 1) noise + silence
        _, audio_with_sil = add_noise_and_silence(audio, Fs)
        self.audio_with_silence = audio_with_sil
        print("Silence Length (samples):", int(Fs * 1))

        # 2) STFT
        S = compute_stft(audio_with_sil)

        # 3) Quantisation
        LEVELS = 8
        _S_abs, S_max, S_norm, S_quant = quantize(S, LEVELS)

        # 4) RLE encode
        data, values, counts = rle_encode(S_quant)
        orig_size = len(data)
        comp_size = len(values) + len(counts)
        print("Original Size:",   orig_size)
        print("Compressed Size:", comp_size)

        # 5) Decode + reconstruct
        S_rec      = rle_decode(values, counts, S_quant.shape)
        signal_rec = reconstruct_signal(S_rec, S_max, audio_with_sil, LEVELS)
        self.signal_rec = signal_rec

        snr = compute_snr(audio_with_sil, signal_rec)
        print(f"SNR = {snr:.4f} dB")

        # store plot payloads
        t  = np.arange(len(audio))         / Fs
        t2 = np.arange(len(audio_with_sil)) / Fs

        self.plot_data = {
            "Original Audio Signal":       ("line",  t,  audio,          "Time", "Amplitude"),
            "Audio with Noise and Silence": ("line",  t2, audio_with_sil, "Time", "Amplitude"),
            "Spectrogram (Manual)":         ("imshow", np.abs(S)),
            "Before Quantization":          ("imshow", S_norm),
            "After Quantization":           ("imshow", S_quant),
            "Original vs Reconstructed":    ("dual",  audio_with_sil, signal_rec),
        }

        # update labels
        self.orig_sz_var.set(f"Original Size: {orig_size}")
        self.comp_sz_var.set(f"Compressed Size: {comp_size}")
        self.snr_var.set(f"SNR = {snr:.4f} dB")
        self.status_var.set("Pipeline complete.")

        self._switch_plot(self.plot_var.get())

    # ── Plot switching ────────────────────────────────────────────────────────
    def _switch_plot(self, choice):
        if not self.plot_data or choice not in self.plot_data:
            return

        payload = self.plot_data[choice]
        kind    = payload[0]

        if kind == "line":
            _, t, sig, xlabel, ylabel = payload
            self._draw_line(t, sig, choice, xlabel, ylabel)
        elif kind == "imshow":
            _, data = payload
            self._draw_imshow(data, choice)
        elif kind == "dual":
            _, orig, rec = payload
            self._draw_dual(orig, rec)

    # ── Drawing helpers ───────────────────────────────────────────────────────
    def _clear_fig(self):
        """Remove all axes + colorbars cleanly."""
        self.fig.clf()


def _draw_line(self, t, signal, title, xlabel, ylabel):
        self._clear_fig()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("white")
        ax.plot(t, signal, color="#0072BD", linewidth=0.6)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        self.fig.tight_layout(pad=1.8)
        self.canvas.draw()
        self.ax = ax

    def _draw_imshow(self, data, title):
        self._clear_fig()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("white")
        im = ax.imshow(data, aspect="auto", origin="lower")
        ax.set_title(title, fontweight="bold", fontsize=10)
        self.fig.colorbar(im, ax=ax)
        self.fig.tight_layout(pad=1.8)
        self.canvas.draw()
        self.ax = ax

    def _draw_dual(self, orig, rec):
        self._clear_fig()
        ax1 = self.fig.add_subplot(2, 1, 1)
        ax2 = self.fig.add_subplot(2, 1, 2)
        for ax in (ax1, ax2):
            ax.set_facecolor("white")

        ax1.plot(orig, color="#0072BD", linewidth=0.5)
        ax1.set_title("Original Audio", fontsize=9)

        ax2.plot(rec, color="#D45500", linewidth=0.5)
        ax2.set_title("Reconstructed Audio", fontsize=9)

        self.fig.tight_layout(pad=1.8)
        self.canvas.draw()
        self.ax = ax1   # keep a reference (ax1 is primary)


# ══════════════════════════════════════════════════════════════════════════════
if name == "main":
    root = tk.Tk()
    DSPProjectGUI(root)
    root.mainloop()
