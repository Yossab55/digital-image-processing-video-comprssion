"""
Video Compression Pipeline - Complete GUI with Side-by-Side Video Display
Suez Canal University - Faculty of Computers and Informatics
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
import numpy as np
from video_compression import VideoCompressionEngine
import threading
from PIL import Image, ImageTk
import cv2

class VideoCompressionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎥 Video Compression Pipeline - Suez Canal University")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#1e1e2e')
        
        self.engine = VideoCompressionEngine()
        self.current_step = 1
        self.current_frame_idx = 0
        self.setup_styles()
        self.setup_gui()
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
    def setup_gui(self):
        # Header
        header = tk.Frame(self.root, bg='#2d2d3d', height=60)
        header.pack(fill='x', padx=10, pady=5)
        header.pack_propagate(False)
        
        title = tk.Label(header, text="VIDEO COMPRESSION PIPELINE", font=('Arial', 16, 'bold'),
                         bg='#2d2d3d', fg='#e5c07b')
        title.pack(side='left', padx=20, pady=10)
        
        subtitle = tk.Label(header, text="DCT | Motion Estimation | Huffman Coding | PSNR Analysis",
                            font=('Arial', 9), bg='#2d2d3d', fg='#abb2bf')
        subtitle.pack(side='left', padx=10)
        
        # Control Panel
        control_panel = tk.Frame(self.root, bg='#2d2d3d', height=50)
        control_panel.pack(fill='x', padx=10, pady=5)
        control_panel.pack_propagate(False)
        
        btn_style = {'font': ('Arial', 9, 'bold'), 'bg': '#61afef', 'fg': '#1e1e2e',
                     'activebackground': '#98c379', 'relief': 'flat', 'padx': 12, 'pady': 5}
        
        self.load_btn = tk.Button(control_panel, text="📁 LOAD VIDEO", command=self.load_video, **btn_style)
        self.load_btn.pack(side='left', padx=10)
        
        self.compress_btn = tk.Button(control_panel, text="⚡ COMPRESS", command=self.compress_video,
                                       **btn_style, state='disabled')
        self.compress_btn.pack(side='left', padx=10)
        
        # Parameters
        param_frame = tk.Frame(control_panel, bg='#2d2d3d')
        param_frame.pack(side='left', padx=15)
        
        tk.Label(param_frame, text="GOP:", bg='#2d2d3d', fg='#abb2bf', font=('Arial', 9)).pack(side='left')
        self.gop_var = tk.IntVar(value=10)
        tk.Spinbox(param_frame, from_=1, to=30, textvariable=self.gop_var, width=3,
                   bg='#3e3e4e', fg='white', relief='flat').pack(side='left', padx=3)
        
        tk.Label(param_frame, text="Q:", bg='#2d2d3d', fg='#abb2bf', font=('Arial', 9)).pack(side='left', padx=(8,0))
        self.q_var = tk.IntVar(value=15)
        tk.Spinbox(param_frame, from_=5, to=50, textvariable=self.q_var, width=3,
                   bg='#3e3e4e', fg='white', relief='flat').pack(side='left', padx=3)
        
        tk.Label(param_frame, text="Max Frames:", bg='#2d2d3d', fg='#abb2bf', font=('Arial', 9)).pack(side='left', padx=(8,0))
        self.max_frames_var = tk.IntVar(value=60)
        tk.Spinbox(param_frame, from_=10, to=200, textvariable=self.max_frames_var, width=4,
                   bg='#3e3e4e', fg='white', relief='flat').pack(side='left', padx=3)
        
        # Progress
        self.progress = ttk.Progressbar(control_panel, mode='determinate', length=250)
        self.progress.pack(side='left', padx=15)
        
        self.status_label = tk.Label(control_panel, text="✅ READY", bg='#2d2d3d', fg='#98c379',
                                      font=('Arial', 9, 'bold'))
        self.status_label.pack(side='left', padx=5)
        
        self.time_label = tk.Label(control_panel, text="", bg='#2d2d3d', fg='#e5c07b', font=('Arial', 8))
        self.time_label.pack(side='left', padx=5)
        
        # Metrics Panel
        metrics_frame = tk.Frame(self.root, bg='#2d2d3d', height=40)
        metrics_frame.pack(fill='x', padx=10, pady=3)
        metrics_frame.pack_propagate(False)
        
        self.cr_label = tk.Label(metrics_frame, text="📊 Compression Ratio: --", bg='#2d2d3d',
                                  fg='#e5c07b', font=('Arial', 10, 'bold'))
        self.cr_label.pack(side='left', padx=15)
        
        self.psnr_label = tk.Label(metrics_frame, text="🎯 Average PSNR: -- dB", bg='#2d2d3d',
                                    fg='#e5c07b', font=('Arial', 10, 'bold'))
        self.psnr_label.pack(side='left', padx=15)
        
        self.size_label = tk.Label(metrics_frame, text="💾 Size: --", bg='#2d2d3d',
                                    fg='#61afef', font=('Arial', 9))
        self.size_label.pack(side='left', padx=15)
        
        # === SIDE-BY-SIDE VIDEO DISPLAY ===
        video_display_frame = tk.Frame(self.root, bg='#1e1e2e', height=280)
        video_display_frame.pack(fill='x', padx=10, pady=5)
        video_display_frame.pack_propagate(False)
        
        # Left: Original Video
        original_frame = tk.LabelFrame(video_display_frame, text="📹 ORIGINAL VIDEO", 
                                        font=('Arial', 10, 'bold'), fg='#61afef', bg='#2d2d3d')
        original_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        self.original_canvas = tk.Canvas(original_frame, bg='#1e1e2e', width=400, height=240)
        self.original_canvas.pack(pady=10, padx=10)
        self.original_label = tk.Label(original_frame, text="No video loaded", bg='#2d2d3d', fg='#abb2bf')
        self.original_label.pack()
        
        # Right: Compressed Video
        compressed_frame = tk.LabelFrame(video_display_frame, text="🗜️ COMPRESSED VIDEO", 
                                          font=('Arial', 10, 'bold'), fg='#98c379', bg='#2d2d3d')
        compressed_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        self.compressed_canvas = tk.Canvas(compressed_frame, bg='#1e1e2e', width=400, height=240)
        self.compressed_canvas.pack(pady=10, padx=10)
        self.compressed_label = tk.Label(compressed_frame, text="Run compression first", bg='#2d2d3d', fg='#abb2bf')
        self.compressed_label.pack()
        
        # Video Navigation
        nav_frame = tk.Frame(video_display_frame, bg='#1e1e2e', height=30)
        nav_frame.pack(fill='x', pady=5)
        
        self.prev_btn = tk.Button(nav_frame, text="◀ PREV", command=self.prev_frame,
                                   bg='#3e3e4e', fg='#abb2bf', relief='flat', padx=10, state='disabled')
        self.prev_btn.pack(side='left', padx=5)
        
        self.frame_counter = tk.Label(nav_frame, text="Frame: 0 / 0", bg='#1e1e2e', fg='#e5c07b', font=('Arial', 9))
        self.frame_counter.pack(side='left', padx=10)
        
        self.next_btn = tk.Button(nav_frame, text="NEXT ▶", command=self.next_frame,
                                   bg='#3e3e4e', fg='#abb2bf', relief='flat', padx=10, state='disabled')
        self.next_btn.pack(side='left', padx=5)
        
        # Step Navigation Buttons
        step_frame = tk.Frame(self.root, bg='#1e1e2e', height=35)
        step_frame.pack(fill='x', padx=10, pady=3)
        
        steps = [
            ('📹 Step 1: Input', 1), ('📊 Step 2: Frames', 2), ('🔬 Step 3: I-frame', 3),
            ('🎯 Step 4: P-frame', 4), ('📈 Step 5: Huffman', 5), ('💾 Step 6: Bitstream', 6), 
            ('✅ Step 7: Results', 7)
        ]
        
        self.step_buttons = []
        for text, step in steps:
            btn = tk.Button(step_frame, text=text, command=lambda s=step: self.show_step(s),
                           font=('Arial', 8), bg='#3e3e4e', fg='#abb2bf',
                           activebackground='#61afef', relief='flat', padx=8, pady=3)
            btn.pack(side='left', padx=2)
            self.step_buttons.append(btn)
        
        # Matplotlib Figure for Step Details
        self.fig = plt.Figure(figsize=(14, 5), facecolor='#1e1e2e', dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=5)
        
        # Show initial welcome
        self.show_welcome()
    
    def show_welcome(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.axis('off')
        ax.set_facecolor('#1e1e2e')
        self.canvas.draw()
    
    def update_status(self, message, progress=None):
        self.status_label.config(text=f"🔄 {message}")
        if progress is not None:
            self.progress['value'] = progress
        self.root.update_idletasks()
    
    def update_video_display(self):
        """Update the side-by-side video display"""
        if self.engine.NF > 0 and self.current_frame_idx < len(self.engine.rgb_frames):
            # Display original frame
            original_img = self.engine.rgb_frames[self.current_frame_idx]
            original_img = cv2.resize(original_img, (400, 240))
            original_img = Image.fromarray(original_img)
            original_photo = ImageTk.PhotoImage(original_img)
            self.original_canvas.create_image(200, 120, image=original_photo, anchor='center')
            self.original_canvas.image = original_photo
            self.original_label.config(text=f"Frame {self.current_frame_idx + 1} of {self.engine.NF}")
            
            # Display compressed frame if available
            if hasattr(self.engine, 'decoded') and len(self.engine.decoded) > 0:
                if self.current_frame_idx < len(self.engine.decoded[0]):
                    compressed_img = self.engine.decoded[:, :, self.current_frame_idx]
                    compressed_img = np.stack([compressed_img]*3, axis=2).astype(np.uint8)
                    compressed_img = cv2.resize(compressed_img, (400, 240))
                    compressed_img = Image.fromarray(compressed_img)
                    compressed_photo = ImageTk.PhotoImage(compressed_img)
                    self.compressed_canvas.create_image(200, 120, image=compressed_photo, anchor='center')
                    self.compressed_canvas.image = compressed_photo
                    
                    psnr_val = self.engine.psnr[self.current_frame_idx] if self.current_frame_idx < len(self.engine.psnr) else 0
                    self.compressed_label.config(text=f"Frame {self.current_frame_idx + 1} | PSNR: {psnr_val:.1f} dB")
                    self.compressed_label.config(fg='#98c379')
                else:
                    self.compressed_label.config(text="Decoding...", fg='#e5c07b')
            else:
                self.compressed_canvas.delete("all")
                self.compressed_canvas.create_text(200, 120, text="Run compression\n to see results", 
                                                    fill='#abb2bf', font=('Arial', 12), anchor='center')
                self.compressed_label.config(text="Not compressed yet", fg='#abb2bf')
            
            # Update frame counter
            self.frame_counter.config(text=f"Frame: {self.current_frame_idx + 1} / {self.engine.NF}")
            
            # Update navigation buttons state
            self.prev_btn.config(state='normal' if self.current_frame_idx > 0 else 'disabled')
            self.next_btn.config(state='normal' if self.current_frame_idx < self.engine.NF - 1 else 'disabled')
    
    def prev_frame(self):
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.update_video_display()
    
    def next_frame(self):
        if self.current_frame_idx < self.engine.NF - 1:
            self.current_frame_idx += 1
            self.update_video_display()
    
    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if path:
            self.update_status("Loading video...")
            self.engine.reset()
            self.current_frame_idx = 0
            max_frames = self.max_frames_var.get()
            
            def load():
                self.engine.load_video(path, max_frames=max_frames, 
                                       progress_callback=lambda c,t: self.root.after(0, lambda: self.update_status(f"Loading... {c}/{t}", (c/t)*100)))
                self.root.after(0, lambda: self.update_status("Video loaded!", 100))
                self.root.after(0, lambda: self.compress_btn.config(state='normal'))
                self.root.after(0, lambda: self.prev_btn.config(state='disabled'))
                self.root.after(0, lambda: self.next_btn.config(state='normal' if self.engine.NF > 1 else 'disabled'))
                self.root.after(0, lambda: self.update_video_display())
                self.root.after(0, lambda: messagebox.showinfo("Success", 
                    f"Loaded {self.engine.NF} frames\nResolution: {self.engine.FW}×{self.engine.FH}\n"
                    f"Max frames limited to: {self.engine.NF}"))
                self.root.after(0, lambda: self.show_step(1))
            
            threading.Thread(target=load, daemon=True).start()
    
    def compress_video(self):
        self.update_status("Compressing... (may take 10-30 seconds)")
        self.compress_btn.config(state='disabled')
        
        def compress():
            try:
                # Compression
                self.engine.compress(
                    gop=self.gop_var.get(), 
                    Q_val=self.q_var.get(), 
                    search=3,
                    progress_callback=lambda c,t: self.root.after(0, lambda: self.update_status(f"Compressing... {c}/{t}", (c/t)*100))
                )
                
                # Decompression
                self.engine.decompress(
                    Q_val=self.q_var.get(),
                    progress_callback=lambda c,t: self.root.after(0, lambda: self.update_status(f"Decompressing... {c}/{t}", (c/t)*100))
                )
                
                # Update metrics
                cr = (self.engine.raw_bytes * 8) / max(self.engine.total_bits, 1)
                avg_psnr = np.mean(self.engine.psnr) if len(self.engine.psnr) > 0 else 0
                
                self.root.after(0, lambda: self.cr_label.config(text=f"📊 Compression Ratio: {cr:.2f}:1"))
                self.root.after(0, lambda: self.psnr_label.config(text=f"🎯 Average PSNR: {avg_psnr:.2f} dB"))
                self.root.after(0, lambda: self.size_label.config(
                    text=f"💾 Raw: {self.engine.raw_bytes/1024:.1f}KB → Comp: {self.engine.total_bits/8/1024:.1f}KB"))
                self.root.after(0, lambda: self.time_label.config(text=f"⏱️ Time: {self.engine.compression_time:.1f}s"))
                self.root.after(0, lambda: self.update_status("Complete!", 100))
                self.root.after(0, lambda: self.update_video_display())
                self.root.after(0, lambda: messagebox.showinfo("Success", 
                    f"Compression Complete!\nRatio: {cr:.2f}:1\nPSNR: {avg_psnr:.2f} dB\nTime: {self.engine.compression_time:.1f}s"))
                self.root.after(0, lambda: self.show_step(7))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.root.after(0, lambda: self.compress_btn.config(state='normal'))
        
        threading.Thread(target=compress, daemon=True).start()
    
    def show_step(self, step):
        self.current_step = step
        self.fig.clear()
        
        # Highlight active step button
        for i, btn in enumerate(self.step_buttons):
            if i + 1 == step:
                btn.config(bg='#61afef', fg='#1e1e2e')
            else:
                btn.config(bg='#3e3e4e', fg='#abb2bf')
        
        if step == 1:
            self.draw_step1()
        elif step == 2:
            self.draw_step2()
        elif step == 3:
            self.draw_step3()
        elif step == 4:
            self.draw_step4()
        elif step == 5:
            self.draw_step5()
        elif step == 6:
            self.draw_step6()
        elif step == 7:
            self.draw_step7()
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def draw_step1(self):
        """Step 1: Video Input"""
        if self.engine.NF == 0:
            self.show_welcome()
            return
        
        n_frames = min(4, self.engine.NF)
        indices = np.linspace(0, self.engine.NF-1, n_frames).astype(int)
        
        for k, idx in enumerate(indices):
            ax = self.fig.add_subplot(2, n_frames, k+1)
            ax.imshow(self.engine.rgb_frames[idx])
            ax.set_title(f"RGB Frame {idx+1}", color='white', fontsize=8)
            ax.axis('off')
            ax.set_facecolor('#1e1e2e')
        
        for k, idx in enumerate(indices):
            ax = self.fig.add_subplot(2, n_frames, n_frames + k + 1)
            ax.imshow(self.engine.yuv_frames[idx][:,:,0], cmap='gray')
            ax.set_title(f"Y-Channel {idx+1}", color='white', fontsize=8)
            ax.axis('off')
            ax.set_facecolor('#1e1e2e')
        
        info_text = f"Total Frames: {self.engine.NF} | Resolution: {self.engine.FW}×{self.engine.FH}"
        self.fig.suptitle("📹 STEP 1: Video Input - RGB Frames vs Y-Channel (Luminance)\n" + info_text,
                         color='#61afef', fontsize=11, fontweight='bold')
        self.fig.patch.set_facecolor('#1e1e2e')
    
    def draw_step2(self):
        """Step 2: Frame Types"""
        if not self.engine.frame_types:
            self.show_welcome()
            return
        
        ax = self.fig.add_subplot(111)
        colors = ['#e06c75' if t == 'I' else '#61afef' for t in self.engine.frame_types]
        
        y = [1 if t == 'I' else 0 for t in self.engine.frame_types]
        ax.bar(range(len(self.engine.frame_types)), y, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['P-Frame', 'I-Frame'], color='white')
        ax.set_xlabel('Frame Index', color='white')
        ax.set_ylabel('Frame Type', color='white')
        ax.set_title(f'Frame Type Decision (GOP = {self.gop_var.get()})', color='#61afef', fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('#2d2d3d')
        ax.tick_params(colors='white')
        
        i_count = self.engine.frame_types.count('I')
        p_count = self.engine.frame_types.count('P')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#e06c75', label='I-Frame (Intra-coded)'),
                          Patch(facecolor='#61afef', label='P-Frame (Predictive)')]
        ax.legend(handles=legend_elements, loc='upper right', facecolor='#2d2d3d', labelcolor='white')
        
        self.fig.suptitle(f"📊 STEP 2: Frame Type Assignment - I-Frames: {i_count} | P-Frames: {p_count}",
                         color='#61afef', fontsize=11, fontweight='bold')
        self.fig.patch.set_facecolor('#1e1e2e')
    
    def draw_step3(self):
        """Step 3: Intra-frame Compression"""
        if not self.engine.comp_data:
            self.show_welcome()
            return
        
        i_frames = [i for i, t in enumerate(self.engine.frame_types) if t == 'I']
        if not i_frames:
            self.fig.text(0.5, 0.5, "No I-frames found", ha='center', va='center', color='white')
            return
        
        fI = i_frames[0]
        Yo = self.engine.yuv_frames[fI][:, :, 0].astype(float)
        
        ax1 = self.fig.add_subplot(1, 3, 1)
        ax1.imshow(Yo, cmap='gray')
        ax1.set_title(f"Original Y-Channel (Frame {fI+1})", color='white', fontsize=9)
        ax1.axis('off')
        
        if self.engine.qY_store[fI] is not None:
            ax2 = self.fig.add_subplot(1, 3, 2)
            im = ax2.imshow(self.engine.qY_store[fI], cmap='plasma')
            ax2.set_title("Quantized DCT Coefficients", color='white', fontsize=9)
            ax2.axis('off')
            self.fig.colorbar(im, ax=ax2)
        
        ax3 = self.fig.add_subplot(1, 3, 3)
        recon = self.engine.recon_iframe(self.engine.comp_data[fI], Yo.shape[0], Yo.shape[1], 
                                          np.ones((8,8)) * self.q_var.get())
        ax3.imshow(recon, cmap='gray')
        ax3.set_title("Reconstructed I-Frame", color='white', fontsize=9)
        ax3.axis('off')
        
        self.fig.suptitle("🔬 STEP 3: Intra-frame Compression - DCT + Quantization + RLE",
                         color='#61afef', fontsize=11, fontweight='bold')
        self.fig.patch.set_facecolor('#1e1e2e')
    
    def draw_step4(self):
        """Step 4: Inter-frame Compression"""
        if not self.engine.mvs_all:
            self.show_welcome()
            return
        
        p_frames = [i for i, t in enumerate(self.engine.frame_types) if t == 'P']
        if not p_frames:
            self.fig.text(0.5, 0.5, "No P-frames found", ha='center', va='center', color='white')
            return
        
        fP = p_frames[0]
        
        ax1 = self.fig.add_subplot(1, 3, 1)
        ax1.imshow(self.engine.yuv_frames[fP][:, :, 0], cmap='gray')
        ax1.set_title(f"Frame {fP+1} - Original Y", color='white', fontsize=9)
        ax1.axis('off')
        
        ax2 = self.fig.add_subplot(1, 3, 2)
        im = ax2.imshow(self.engine.res_all[fP], cmap='RdBu', vmin=-50, vmax=50)
        ax2.set_title("Residual (Prediction Error)", color='white', fontsize=9)
        ax2.axis('off')
        self.fig.colorbar(im, ax=ax2)
        
        ax3 = self.fig.add_subplot(1, 3, 3)
        if self.engine.mvs_all[fP] is not None:
            h_f, w_f = self.engine.res_all[fP].shape
            nbh, nbw = h_f // 8, w_f // 8
            mvs = self.engine.mvs_all[fP]
            mv_dy = mvs[:, 0].reshape(nbh, nbw)
            mv_dx = mvs[:, 1].reshape(nbh, nbw)
            cols, rows = np.meshgrid(np.arange(nbw), np.arange(nbh))
            ax3.quiver(cols, rows, mv_dx, mv_dy, color='#e06c75', scale=20, width=0.005)
            ax3.set_xlim(-0.5, nbw-0.5)
            ax3.set_ylim(nbh-0.5, -0.5)
            ax3.set_title("Motion Vectors", color='white', fontsize=9)
            ax3.set_xlabel("Block Column", color='white')
            ax3.set_ylabel("Block Row", color='white')
            ax3.set_facecolor('#2d2d3d')
            ax3.tick_params(colors='white')
        
        self.fig.suptitle("🎯 STEP 4: Inter-frame Compression - Motion Estimation + Residual",
                         color='#61afef', fontsize=11, fontweight='bold')
        self.fig.patch.set_facecolor('#1e1e2e')
    
    def draw_step5(self):
        """Step 5: Entropy Coding"""
        if len(self.engine.orig_bits) == 0:
            self.show_welcome()
            return
        
        ax1 = self.fig.add_subplot(1, 2, 1)
        x = np.arange(len(self.engine.orig_bits))
        ax1.bar(x, self.engine.orig_bits, alpha=0.7, label='Original (8-bit)', color='#61afef')
        ax1.bar(x, self.engine.enc_bits, alpha=0.7, label='Huffman Encoded', color='#e06c75')
        ax1.set_xlabel('Frame Number', color='white')
        ax1.set_ylabel('Bits', color='white')
        ax1.set_title('Bit Usage per Frame', color='white')
        ax1.legend(facecolor='#2d2d3d', labelcolor='white')
        ax1.grid(True, alpha=0.2)
        ax1.set_facecolor('#2d2d3d')
        ax1.tick_params(colors='white')
        
        ax2 = self.fig.add_subplot(1, 2, 2)
        cr_per_frame = self.engine.orig_bits / np.maximum(self.engine.enc_bits, 1)
        colors = ['#e06c75' if t == 'I' else '#61afef' for t in self.engine.frame_types]
        ax2.bar(x, cr_per_frame, color=colors, alpha=0.8)
        ax2.set_xlabel('Frame Number', color='white')
        ax2.set_ylabel('Compression Ratio', color='white')
        ax2.set_title('Huffman Compression Ratio per Frame', color='white')
        ax2.grid(True, alpha=0.2)
        ax2.set_facecolor('#2d2d3d')
        ax2.tick_params(colors='white')
        
        avg_cr = np.mean(cr_per_frame)
        self.fig.suptitle(f"📈 STEP 5: Entropy Coding - Huffman Compression (Average Ratio: {avg_cr:.2f}:1)",
                         color='#61afef', fontsize=11, fontweight='bold')
        self.fig.patch.set_facecolor('#1e1e2e')
    
    def draw_step6(self):
        """Step 6: Bitstream Formation"""
        if self.engine.total_bits == 0:
            self.show_welcome()
            return
        
        ax1 = self.fig.add_subplot(1, 2, 1)
        frame_bits = [72 + int(self.engine.enc_bits[f]) for f in range(self.engine.NF)]
        colors = ['#e06c75' if self.engine.frame_types[f] == 'I' else '#61afef' for f in range(self.engine.NF)]
        ax1.bar(range(self.engine.NF), frame_bits, color=colors, alpha=0.8)
        ax1.set_xlabel('Frame Index', color='white')
        ax1.set_ylabel('Bits (including 72-bit header)', color='white')
        ax1.set_title('Final Bitstream - Bits per Frame', color='white')
        ax1.grid(True, alpha=0.2)
        ax1.set_facecolor('#2d2d3d')
        ax1.tick_params(colors='white')
        
        ax2 = self.fig.add_subplot(1, 2, 2)
        raw_bytes = self.engine.raw_bytes
        comp_bytes = self.engine.total_bits / 8
        bars = ax2.bar(['Raw Video', 'Compressed'], [raw_bytes, comp_bytes], 
                       color=['#abb2bf', '#98c379'], alpha=0.8)
        ax2.set_ylabel('Size (Bytes)', color='white')
        ax2.set_title(f'Overall Compression: {(raw_bytes*8)/max(self.engine.total_bits,1):.2f}x', color='white')
        ax2.grid(True, alpha=0.2)
        ax2.set_facecolor('#2d2d3d')
        ax2.tick_params(colors='white')
        
        for bar, val in zip(bars, [raw_bytes, comp_bytes]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(raw_bytes, comp_bytes)*0.02,
                    f'{val/1024:.1f} KB', ha='center', va='bottom', color='white', fontsize=8)
        
        total_mb = self.engine.total_bits / 8 / 1024
        self.fig.suptitle(f"💾 STEP 6: Bitstream Formation - Total Size: {total_mb:.2f} KB",
                         color='#61afef', fontsize=11, fontweight='bold')
        self.fig.patch.set_facecolor('#1e1e2e')
    
    def draw_step7(self):
        """Step 7: Evaluation Results"""
        if len(self.engine.psnr) == 0:
            self.show_welcome()
            return
        # PSNR plot
        ax_psnr = self.fig.add_subplot(2, 2, 3)
        ax_psnr.plot(self.engine.psnr, '#61afef', linewidth=2, marker='o', markersize=3)
        ax_psnr.axhline(30, color='#e5c07b', linestyle='--', linewidth=1.5, label='30dB (Acceptable)')
        ax_psnr.axhline(40, color='#98c379', linestyle='--', linewidth=1.5, label='40dB (Excellent)')
        ax_psnr.fill_between(range(len(self.engine.psnr)), 0, self.engine.psnr, alpha=0.3, color='#61afef')
        ax_psnr.set_xlabel('Frame Number', color='white')
        ax_psnr.set_ylabel('PSNR (dB)', color='white')
        ax_psnr.set_title(f'PSNR per Frame (Average: {np.mean(self.engine.psnr):.2f} dB)', color='white')
        ax_psnr.legend(facecolor='#2d2d3d', labelcolor='white', fontsize=7)
        ax_psnr.grid(True, alpha=0.2)
        ax_psnr.set_facecolor('#2d2d3d')
        ax_psnr.tick_params(colors='white')
        
        # Summary
        ax_sum = self.fig.add_subplot(2, 2, 4)
        cr = (self.engine.raw_bytes * 8) / max(self.engine.total_bits, 1)
        avg_psnr = np.mean(self.engine.psnr)
        
        if avg_psnr > 40:
            quality = "EXCELLENT ✓"
            quality_color = '#98c379'
        elif avg_psnr > 35:
            quality = "GOOD ✓"
            quality_color = '#e5c07b'
        elif avg_psnr > 30:
            quality = "ACCEPTABLE ✓"
            quality_color = '#61afef'
        else:
            quality = "POOR ✗"
            quality_color = "#fbfbfb"
        
        summary_text = f"""
╔══════════════════════════════════════╗
║     FINAL EVALUATION SUMMARY         ║
╠══════════════════════════════════════╣
║  Average PSNR: {avg_psnr:.2f} dB                  ║
║  Compression Ratio: {cr:.2f}:1                    ║
║  Total Frames: {self.engine.NF}                         ║
║  Resolution: {self.engine.FW}×{self.engine.FH}                       ║
║  Raw Size: {self.engine.raw_bytes/1024:.1f} KB                    ║
║  Compressed: {self.engine.total_bits/8/1024:.1f} KB                   ║
║  Compression Time: {self.engine.compression_time:.1f}s                ║
╠══════════════════════════════════════╣
║  QUALITY: {quality:<30} ║
╚══════════════════════════════════════╝
        """
        
        ax_sum.text(0.5, 0.5, summary_text, transform=ax_sum.transAxes, fontsize=8,
                   verticalalignment='center', horizontalalignment='center',
                   fontfamily='monospace', color=quality_color, fontweight='bold')
        ax_sum.axis('off')
        ax_sum.set_facecolor('#1e1e2e')
        
        self.fig.suptitle("🎯 STEP 7: Testing & Evaluation - Original vs Decoded with PSNR Analysis",
                         color='#61afef', fontsize=11, fontweight='bold')
        self.fig.patch.set_facecolor('#1e1e2e')


def main():
    root = tk.Tk()
    app = VideoCompressionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()