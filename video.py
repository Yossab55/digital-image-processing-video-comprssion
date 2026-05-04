"""
Video Compression Pipeline - Steps 1 to 7
Suez Canal University | Faculty of Computers and Informatics
=========================================================
Implements a simplified H.264-style codec:
  Step 1 - Video Input & YCbCr Conversion
  Step 2 - Frame Type Assignment (I / P frames)
  Step 3 - Intra-frame Compression  (DCT + Quantization + RLE)
  Step 4 - Inter-frame Compression  (Block Matching + Residuals)
  Step 5 - Entropy Coding           (Manual Huffman)
  Step 6 - Bitstream Formation
  Step 7 - Decoding & Evaluation    (PSNR + Compression Ratio)

All 7 steps are visualised in ONE combined figure at the end.

Install: pip install opencv-python numpy scipy matplotlib
"""

import cv2, numpy as np, matplotlib.pyplot as plt, matplotlib.patches as mpatches
from scipy.fft import dctn, idctn
from collections import Counter
import heapq, os
import matplotlib
matplotlib.use('Agg')  # non-GUI backend

# ─────────────────────────────────────────────
# CONFIGURATION  (tweak these to explore)
# ─────────────────────────────────────────────
VIDEO_PATH   = "./Thumbs-up.mp4"   # Supply your own video or leave as-is for synthetic
BLOCK_SIZE   = 8                # Standard JPEG/H.264 macro-block size
Q_SCALAR     = 10.0             # Quantisation step: larger = more compression, less quality
Q_MAT        = np.ones((8,8)) * Q_SCALAR   # Flat quantisation matrix (uniform quality)
SEARCH_RANGE = 4                # Motion-vector search radius in pixels (±4 px)
I_INTERVAL   = 10               # Insert an I-frame every N frames (like Group-of-Pictures)
Q_STEP_HUFF  = 10               # Quantisation step applied before Huffman symbol mapping


# ─────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────

def zigzag_scan(block):
    """Re-order an 8×8 DCT block into a 1-D array using zig-zag traversal.
    This groups the low-frequency (important) coefficients at the start
    and high-frequency (near-zero after quantisation) ones at the end,
    maximising the run-length of zeros for better RLE compression."""
    order = (np.array([
         1, 2, 6, 7,15,16,28,29,
         3, 5, 8,14,17,27,30,43,
         4, 9,13,18,26,31,42,44,
        10,12,19,25,32,41,45,54,
        11,20,24,33,40,46,53,55,
        21,23,34,39,47,52,56,61,
        22,35,38,48,51,57,60,62,
        36,37,49,50,58,59,63,64
    ]).reshape(8,8) - 1)
    flat = np.zeros(64)
    for r in range(8):
        for c in range(8):
            flat[order[r,c]] = block[r,c]
    return flat


def rle_encode(arr):
    """Run-Length Encoding: collapse consecutive identical values into (value, count) pairs.
    Especially effective after zig-zag scan because long runs of zeros appear at the tail."""
    enc, count = [], 1
    for i in range(1, len(arr)):
        if arr[i] == arr[i-1]: count += 1
        else: enc.append((arr[i-1], count)); count = 1
    enc.append((arr[-1], count))
    return enc


def rle_decode(enc):
    """Reverse RLE: expand (value, count) pairs back to a flat array."""
    arr = []
    for val, cnt in enc:
        arr.extend([val]*int(cnt))
    return np.array(arr)


def reconstruct_iframe(compressed_frame, Q, h, w, bs=8):
    """Reconstruct the Y-channel of an I-frame from its RLE+DCT compressed blocks.
    Process: RLE decode → inverse zig-zag (implicit via reshape) → dequantise → IDCT."""
    Y_rec = np.zeros((h, w))
    for idx, (row, col) in enumerate(((r,c) for r in range(h//bs) for c in range(w//bs))):
        if idx >= len(compressed_frame): break
        flat = rle_decode(compressed_frame[idx])
        flat = np.pad(flat, (0, max(0, 64-len(flat))))
        block = idctn((flat[:64].reshape(8,8) * Q), norm='ortho')
        Y_rec[row*bs:(row+1)*bs, col*bs:(col+1)*bs] = block
    return np.clip(Y_rec, 0, 255)


# ─────────────────────────────────────────────
# HUFFMAN CODING  (pure Python, no library)
# ─────────────────────────────────────────────

class _HNode:
    """Internal Huffman tree node."""
    def __init__(self, sym, p): self.sym=sym; self.p=p; self.L=self.R=None
    def __lt__(self, o): return self.p < o.p

def build_huffman(symbols, probs):
    """Build a Huffman code dictionary.
    Greedy min-heap strategy: repeatedly merge the two lowest-probability nodes
    until only the root remains. Frequent symbols → short codes; rare → long codes."""
    if len(symbols)==1: return {symbols[0]:'0'}
    heap = [_HNode(s,p) for s,p in zip(symbols,probs)]
    heapq.heapify(heap)
    while len(heap)>1:
        L,R = heapq.heappop(heap), heapq.heappop(heap)
        m = _HNode(None, L.p+R.p); m.L=L; m.R=R
        heapq.heappush(heap, m)
    codes = {}
    def _walk(node, code):
        if node is None: return
        if node.sym is not None: codes[node.sym] = code or '0'; return
        _walk(node.L, code+'0'); _walk(node.R, code+'1')
    _walk(heap[0],'')
    return codes

def huff_encode(syms, d):
    return ''.join(d.get(s,'') for s in syms)

def huff_decode(encoded, d):
    """Decode a binary string using the reverse Huffman lookup table."""
    rev = {v:k for k,v in d.items()}
    maxlen = max(len(c) for c in rev)
    out, i = [], 0
    while i < len(encoded):
        for l in range(1, min(maxlen, len(encoded)-i)+1):
            cw = encoded[i:i+l]
            if cw in rev: out.append(rev[cw]); i+=l; break
        else: break
    return np.array(out)


# ══════════════════════════════════════════════════════════════════
# STEP 1 — VIDEO INPUT & COLOUR SPACE CONVERSION
# ══════════════════════════════════════════════════════════════════
print("► STEP 1  Loading video & converting to YCbCr ...")

if not os.path.exists(VIDEO_PATH):
    # Synthetic fallback: a coloured rectangle that moves across the frame
    print("  (No video file found — using synthetic 30-frame test video)")
    H, W, NF = 120, 160, 30
    frames_rgb = []
    for f in range(NF):
        img = np.zeros((H,W,3), dtype=np.uint8)
        off = (f*3) % (W-40)
        img[30:70, off:off+40] = [180,100,60]
        img += np.random.randint(0,8,img.shape, dtype=np.uint8)
        frames_rgb.append(img)
else:
    cap = cv2.VideoCapture(VIDEO_PATH)
    frames_rgb = []
    while True:
        ret, f = cap.read()
        if not ret: break
        frames_rgb.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()

# Convert each RGB frame to YCrCb  (Y = luma/brightness, used for compression)
yuv_frames = [cv2.cvtColor(fr, cv2.COLOR_RGB2YCrCb) for fr in frames_rgb]
NF, H, W = len(frames_rgb), frames_rgb[0].shape[0], frames_rgb[0].shape[1]
print(f"  {NF} frames  |  {H}×{W} px")


# ══════════════════════════════════════════════════════════════════
# STEP 2 — FRAME TYPE ASSIGNMENT  (I-frame / P-frame)
# ══════════════════════════════════════════════════════════════════
print("► STEP 2  Assigning I / P frame types ...")

# I-frame (Intra): self-contained, compressed independently like a JPEG.
# P-frame (Predictive): stores only the difference from the previous frame.
frame_types = ['I' if f % I_INTERVAL == 0 else 'P' for f in range(NF)]
print(f"  I-frames: {frame_types.count('I')}   P-frames: {frame_types.count('P')}")


# ══════════════════════════════════════════════════════════════════
# STEP 3 — INTRA-FRAME COMPRESSION  (DCT + Quantise + Zig-Zag + RLE)
# ══════════════════════════════════════════════════════════════════
print("► STEP 3  Compressing I-frames (DCT → Quantise → Zig-Zag → RLE) ...")

compressed_data    = [None]*NF   # Stores RLE-encoded blocks per frame
iframe_Y_quantized = [None]*NF   # Stores quantised DCT maps for display

for f in range(NF):
    if frame_types[f] != 'I': continue
    Y = yuv_frames[f][:,:,0].astype(float)
    h, w = Y.shape
    cf, qY = [], np.zeros((h,w))
    for r in range(0, h-BLOCK_SIZE+1, BLOCK_SIZE):
        for c in range(0, w-BLOCK_SIZE+1, BLOCK_SIZE):
            blk   = Y[r:r+8, c:c+8]
            dct_b = dctn(blk, norm='ortho')           # 2-D DCT (energy compaction)
            q_b   = np.round(dct_b / Q_MAT)           # Quantise: reduces precision of HF
            cf.append(rle_encode(zigzag_scan(q_b)))   # Zig-zag then RLE
            qY[r:r+8, c:c+8] = q_b
    compressed_data[f] = cf
    iframe_Y_quantized[f] = qY
    print(f"  Frame {f+1:3d} [I]  {len(cf)} blocks compressed")


# ══════════════════════════════════════════════════════════════════
# STEP 4 — INTER-FRAME COMPRESSION  (Block Matching + Residuals)
# ══════════════════════════════════════════════════════════════════
print("► STEP 4  Compressing P-frames (Block Matching → Residuals) ...")

motion_vectors = [None]*NF
residuals      = [None]*NF

for f in range(NF):
    Y_cur = yuv_frames[f][:,:,0].astype(float)
    h, w  = Y_cur.shape

    if frame_types[f] == 'I':
        # I-frames: reconstruct immediately so P-frames can use them as reference
        motion_vectors[f] = None
        residuals[f] = reconstruct_iframe(compressed_data[f], Q_MAT, h, w, BLOCK_SIZE)
        continue

    # For P-frames: search the previous frame for the best-matching block
    # Motion Vector (dy,dx): tells the decoder where the block "came from"
    Y_ref = yuv_frames[f-1][:,:,0].astype(float)
    nbh, nbw = h//BLOCK_SIZE, w//BLOCK_SIZE
    mvs  = np.zeros((nbh*nbw, 2), dtype=int)
    res  = np.zeros((h, w))

    for idx, (row, col) in enumerate(((r,c) for r in range(nbh) for c in range(nbw))):
        rs, cs = row*BLOCK_SIZE, col*BLOCK_SIZE
        blk = Y_cur[rs:rs+8, cs:cs+8]
        best_mad, bdy, bdx = np.inf, 0, 0
        for dy in range(-SEARCH_RANGE, SEARCH_RANGE+1):
            for dx in range(-SEARCH_RANGE, SEARCH_RANGE+1):
                rr, rc = rs+dy, cs+dx
                if rr<0 or rc<0 or rr+8>h or rc+8>w: continue
                mad = np.mean(np.abs(blk - Y_ref[rr:rr+8, rc:rc+8]))
                if mad < best_mad: best_mad,bdy,bdx = mad,dy,dx
        mvs[idx] = [bdy, bdx]
        rr = np.clip(rs+bdy, 0, h-8); rc = np.clip(cs+bdx, 0, w-8)
        res[rs:rs+8, cs:cs+8] = blk - Y_ref[rr:rr+8, rc:rc+8]  # residual error

    motion_vectors[f] = mvs
    residuals[f] = res
    print(f"  Frame {f+1:3d} [P]  Mean MV magnitude: {np.mean(np.sqrt(np.sum(mvs**2,1))):.2f} px")


# ══════════════════════════════════════════════════════════════════
# STEP 5 — ENTROPY CODING  (Manual Huffman)
# ══════════════════════════════════════════════════════════════════
print("► STEP 5  Entropy coding with Huffman ...")

enc_streams  = [None]*NF
huff_dicts   = [None]*NF
orig_bits    = np.zeros(NF)
enc_bits     = np.zeros(NF)

for f in range(NF):
    # Flatten all values that need to be transmitted for this frame
    if frame_types[f] == 'I':
        syms = np.round(residuals[f] / Q_STEP_HUFF).astype(int).flatten()
    else:
        mv_flat  = motion_vectors[f].flatten()
        res_flat = np.round(residuals[f].flatten() / Q_STEP_HUFF).astype(int)
        syms     = np.concatenate([mv_flat, res_flat])

    uniq, cnt = np.unique(syms, return_counts=True)
    hd = build_huffman(uniq.tolist(), (cnt/cnt.sum()).tolist())
    enc = huff_encode(syms.tolist(), hd)

    enc_streams[f] = enc;  huff_dicts[f] = hd
    orig_bits[f]   = len(syms)*8          # 8 bits/symbol uncompressed
    enc_bits[f]    = len(enc)             # variable-length after Huffman
    print(f"  Frame {f+1:3d} [{frame_types[f]}]  {int(orig_bits[f]):7d}→{int(enc_bits[f]):7d} bits  ratio={orig_bits[f]/max(enc_bits[f],1):.2f}×")


# ══════════════════════════════════════════════════════════════════
# STEP 6 — BITSTREAM FORMATION
# ══════════════════════════════════════════════════════════════════
print("► STEP 6  Forming bitstream ...")

HEADER_BITS = 72   # 32b frame-index + 8b type + 32b payload-size  (like NAL units in H.264)
bitstream   = {'header': dict(num_frames=NF,H=H,W=W,bs=BLOCK_SIZE,q=Q_STEP_HUFF), 'frames':[]}
total_bits  = 0

for f in range(NF):
    fb = HEADER_BITS + int(enc_bits[f])
    bitstream['frames'].append(dict(
        index=f, type=frame_types[f],
        enc=enc_streams[f], hd=huff_dicts[f], total=fb))
    total_bits += fb

raw_bytes = H * W * NF   # Raw = 1 byte per pixel (Y channel only)
print(f"  Compressed : {total_bits/8/1024:.1f} KB   Raw: {raw_bytes/1024:.1f} KB   CR: {raw_bytes*8/total_bits:.2f}×")


# ══════════════════════════════════════════════════════════════════
# STEP 7 — DECODING & EVALUATION  (PSNR + Compression Ratio)
# ══════════════════════════════════════════════════════════════════
print("► STEP 7  Decoding and evaluating PSNR ...")

# PSNR (Peak Signal-to-Noise Ratio):
#   > 40 dB  → Excellent (near-lossless perceptually)
#   30–40 dB → Acceptable (visible but minor artefacts)
#   < 30 dB  → Noticeable quality loss

decoded = np.zeros((H, W, NF))
psnr    = np.zeros(NF)

for f in range(NF):
    fd  = bitstream['frames'][f]
    sym = huff_decode(fd['enc'], fd['hd'])
    nbh, nbw = H//BLOCK_SIZE, W//BLOCK_SIZE

    if fd['type'] == 'I':
        npix = nbh*nbw*BLOCK_SIZE**2
        rec  = np.pad(sym, (0,max(0,npix-len(sym))))[:npix].astype(float) * Q_STEP_HUFF
        decoded[:,:,f] = np.clip(rec.reshape(H,W), 0, 255)

    else:
        nb = nbh*nbw
        nmv = nb*2
        npix = nb*BLOCK_SIZE**2
        sym  = np.pad(sym,(0,max(0,nmv+npix-len(sym))))
        mv_d = sym[:nmv].reshape(nb,2)
        res  = sym[nmv:nmv+npix].astype(float)*Q_STEP_HUFF
        res_img = res.reshape(H,W)
        ref  = decoded[:,:,f-1]
        pred = np.zeros((H,W))
        for idx,(row,col) in enumerate(((r,c) for r in range(nbh) for c in range(nbw))):
            rs,cs = row*BLOCK_SIZE, col*BLOCK_SIZE
            dy,dx = int(mv_d[idx,0]), int(mv_d[idx,1])
            rr = np.clip(rs+dy,0,H-8); rc = np.clip(cs+dx,0,W-8)
            pred[rs:rs+8,cs:cs+8] = ref[rr:rr+8,rc:rc+8]
        decoded[:,:,f] = np.clip(pred+res_img, 0, 255)

    mse     = np.mean((yuv_frames[f][:,:,0].astype(float) - decoded[:,:,f])**2)
    psnr[f] = 10*np.log10(255**2 / max(mse,1e-10))
    print(f"  Frame {f+1:3d} [{fd['type']}]  PSNR = {psnr[f]:.1f} dB")

overall_cr   = raw_bytes*8/total_bits
overall_psnr = np.mean(psnr)
print(f"\n  ═══ SUMMARY ═══  PSNR={overall_psnr:.1f} dB   CR={overall_cr:.2f}×")


# ══════════════════════════════════════════════════════════════════
# COMBINED OUTPUT FIGURE  —  All 7 Steps on One Page
# ══════════════════════════════════════════════════════════════════
print("\n► Building combined figure ...")

# Indices for representative frames
si   = np.round(np.linspace(0,NF-1,4)).astype(int)   # 4 evenly-spaced frames
fi   = next(f for f,t in enumerate(frame_types) if t=='I')   # first I-frame
fp   = next(f for f,t in enumerate(frame_types) if t=='P')   # first P-frame
nbh0 = H//BLOCK_SIZE; nbw0 = W//BLOCK_SIZE
mvs0 = motion_vectors[fp]

fig = plt.figure(figsize=(22, 28), facecolor='#0d1117')
fig.suptitle(
    "Video Compression Pipeline  —  Steps 1–7\n"
    "Suez Canal University · Faculty of Computers and Informatics",
    fontsize=15, color='white', fontweight='bold', y=0.995)

LABEL_STYLE = dict(fontsize=8, color='white')
TITLE_STYLE = dict(fontsize=9, color='#58a6ff', fontweight='bold')

def ax_off(ax, title='', img=None, cmap='gray', vmin=None, vmax=None):
    if img is not None:
        kw = {}
        if vmin is not None: kw['vmin']=vmin
        if vmax is not None: kw['vmax']=vmax
        ax.imshow(img, cmap=cmap, **kw)
    ax.axis('off')
    if title: ax.set_title(title, **TITLE_STYLE)
    ax.set_facecolor('#0d1117')


# ── Row 1: Step 1  (4 RGB + 4 Y-channel) ──────────────────────
for k,idx in enumerate(si):
    ax = fig.add_subplot(8, 8, k+1)
    ax_off(ax, f"RGB frame {idx+1}", frames_rgb[idx])
    ax = fig.add_subplot(8, 8, k+5)
    ax_off(ax, f"Y-channel {idx+1}", yuv_frames[idx][:,:,0], cmap='gray', vmin=0, vmax=255)

# Step label
fig.text(0.01, 0.945, "STEP 1\nVideo Input\n& YCbCr", fontsize=8,
         color='#3fb950', va='top', fontweight='bold')
fig.text(0.99, 0.945,
         "Each RGB frame is converted to YCbCr.\n"
         "The Y (luma) channel carries perceptual brightness\n"
         "and is the primary channel for compression.",
         fontsize=7, color='#8b949e', va='top', ha='right')


# ── Row 2: Step 2 (frame-type stem) + Step 3 (I-frame recon) ──
ax2 = fig.add_subplot(8, 4, 5)
ax2.set_facecolor('#0d1117')
type_n = [1 if t=='I' else 0 for t in frame_types]
ax2.stem(range(NF), type_n, markerfmt='C0o', linefmt='C0-', basefmt='#444')
ax2.set_yticks([0,1]); ax2.set_yticklabels(['P','I'], color='white', fontsize=8)
ax2.set_xlabel('Frame index', color='#8b949e', fontsize=7)
ax2.set_title("STEP 2 — Frame Type Assignment\n"
              "I-frame every 10th (self-contained); P-frames predict from prev.", **TITLE_STYLE)
ax2.tick_params(colors='#8b949e', labelsize=7); ax2.grid(alpha=0.2)
for sp in ax2.spines.values(): sp.set_color('#30363d')

Y_orig  = yuv_frames[fi][:,:,0].astype(float)
Y_recon = reconstruct_iframe(compressed_data[fi], Q_MAT, H, W, BLOCK_SIZE)

ax3a = fig.add_subplot(8, 4, 6)
ax_off(ax3a, f"STEP 3 — Original Y (Frame {fi+1})\nDCT+Quant+RLE applied to each 8×8 block", Y_orig, vmin=0, vmax=255)

ax3b = fig.add_subplot(8, 4, 7)
ax_off(ax3b, "Quantised DCT coefficients\n(sparse → efficient RLE)", iframe_Y_quantized[fi])

ax3c = fig.add_subplot(8, 4, 8)
ax_off(ax3c, f"Reconstructed I-frame\nPSNR={psnr[fi]:.1f} dB", Y_recon, vmin=0, vmax=255)


# ── Row 3: Step 4 (residual + motion vectors) ──────────────────
ax4a = fig.add_subplot(8, 4, 9)
ax_off(ax4a, f"STEP 4 — P-frame Residual (Frame {fp+1})\nCurrent − Predicted (block matching, ±{SEARCH_RANGE}px)",
       residuals[fp], cmap='gray')

ax4b = fig.add_subplot(8, 4, 10)
ax4b.set_facecolor('#0d1117')
cols,rows = np.meshgrid(np.arange(1,nbw0+1), np.arange(1,nbh0+1))
ax4b.quiver(cols, rows, mvs0[:,1].reshape(nbh0,nbw0), mvs0[:,0].reshape(nbh0,nbw0),
            color='#f78166', scale=40)
ax4b.set_xlim(0,nbw0+1); ax4b.set_ylim(0,nbh0+1); ax4b.invert_yaxis()
ax4b.set_title(f"Motion Vectors (Frame {fp+1})\nEach arrow = displacement of one 8×8 block", **TITLE_STYLE)
ax4b.tick_params(colors='#8b949e', labelsize=7)
for sp in ax4b.spines.values(): sp.set_color('#30363d')

# Residual histogram
ax4c = fig.add_subplot(8, 4, 11)
ax4c.set_facecolor('#0d1117')
ax4c.hist(residuals[fp].flatten(), bins=60, color='#58a6ff', edgecolor='none', density=True)
ax4c.set_title("Residual Distribution\nPeaked near 0 → compressible", **TITLE_STYLE)
ax4c.tick_params(colors='#8b949e', labelsize=7)
for sp in ax4c.spines.values(): sp.set_color('#30363d')

# Reference frame used for prediction
ax4d = fig.add_subplot(8, 4, 12)
ax_off(ax4d, f"Reference frame (Frame {fp})\nUsed as predictor for Frame {fp+1}",
       yuv_frames[fp-1][:,:,0], vmin=0, vmax=255)


# ── Row 4: Step 5 (Huffman bit usage + ratio) ──────────────────
ax5a = fig.add_subplot(8, 2, 9)
ax5a.set_facecolor('#0d1117')
x = np.arange(NF)
ax5a.bar(x, orig_bits,  color='#388bfd', alpha=0.6, label='Uncompressed (8 bit/sym)')
ax5a.bar(x, enc_bits,   color='#f85149', alpha=0.9, label='Huffman encoded')
ax5a.set_xlabel('Frame index', color='#8b949e', fontsize=7)
ax5a.set_ylabel('Bits', color='#8b949e', fontsize=7)
ax5a.set_title("STEP 5 — Entropy Coding (Huffman)\n"
               "Frequent symbols → short codes; rare → long codes", **TITLE_STYLE)
ax5a.legend(fontsize=7, facecolor='#161b22', labelcolor='white')
ax5a.tick_params(colors='#8b949e', labelsize=7)
for sp in ax5a.spines.values(): sp.set_color('#30363d')

ax5b = fig.add_subplot(8, 2, 10)
ax5b.set_facecolor('#0d1117')
cr_frame = orig_bits / np.maximum(enc_bits,1)
ax5b.plot(cr_frame, 'o-', color='#3fb950', linewidth=1.5, markersize=4)
ax5b.set_xlabel('Frame index', color='#8b949e', fontsize=7)
ax5b.set_ylabel('Compression ratio', color='#8b949e', fontsize=7)
ax5b.set_title("Huffman Compression Ratio per Frame\nI-frames: larger; P-frames: smaller residuals", **TITLE_STYLE)
ax5b.tick_params(colors='#8b949e', labelsize=7)
ax5b.grid(alpha=0.2)
for sp in ax5b.spines.values(): sp.set_color('#30363d')


# ── Row 5: Step 6 (bitstream bar + pie) ────────────────────────
frame_bits_arr = [bitstream['frames'][f]['total'] for f in range(NF)]
bar_clr = ['#f85149' if t=='I' else '#388bfd' for t in frame_types]

ax6a = fig.add_subplot(8, 2, 11)
ax6a.set_facecolor('#0d1117')
ax6a.bar(range(NF), frame_bits_arr, color=bar_clr)
ip = mpatches.Patch(color='#f85149', label='I-frame')
pp = mpatches.Patch(color='#388bfd', label='P-frame')
ax6a.legend(handles=[ip,pp], fontsize=7, facecolor='#161b22', labelcolor='white')
ax6a.set_xlabel('Frame index', color='#8b949e', fontsize=7)
ax6a.set_ylabel('Bits (incl. 72-bit header)', color='#8b949e', fontsize=7)
ax6a.set_title(f"STEP 6 — Bitstream Formation\n"
               f"Each frame: 72-bit NAL header + Huffman payload  (Total: {total_bits/8/1024:.1f} KB)", **TITLE_STYLE)
ax6a.tick_params(colors='#8b949e', labelsize=7)
for sp in ax6a.spines.values(): sp.set_color('#30363d')

ax6b = fig.add_subplot(8, 2, 12)
ax6b.set_facecolor('#0d1117')
saved_kb = max(raw_bytes - total_bits/8, 0)
ax6b.pie(
    [total_bits/8, saved_kb],
    labels=[f"Compressed\n{total_bits/8/1024:.1f} KB", f"Saved\n{saved_kb/1024:.1f} KB"],
    colors=['#f85149','#3fb950'], autopct='%1.1f%%',
    textprops=dict(color='white', fontsize=8))
ax6b.set_title(f"Overall Compression Ratio: {overall_cr:.2f}×\n(Y-channel only)", **TITLE_STYLE)


# ── Row 6–7: Step 7 (original vs decoded + PSNR curve) ─────────
show_f = list(dict.fromkeys([0,1,fi,fp,NF-1]))[:4]

for k,f in enumerate(show_f):
    ax = fig.add_subplot(8, 8, 8*6 + k + 1)
    ax_off(ax, f"Orig F{f+1} [{frame_types[f]}]", yuv_frames[f][:,:,0], vmin=0, vmax=255)
    ax = fig.add_subplot(8, 8, 8*6 + k + 5)
    ax_off(ax, f"Decoded F{f+1}\n{psnr[f]:.1f} dB", decoded[:,:,f], vmin=0, vmax=255)

fig.text(0.01, 0.205, "STEP 7\nOriginal vs\nDecoded", fontsize=8,
         color='#3fb950', va='top', fontweight='bold')

ax7 = fig.add_subplot(8, 1, 8)
ax7.set_facecolor('#0d1117')
ax7.plot(psnr, 'o-', color='#58a6ff', linewidth=1.5, markersize=4, label='PSNR per frame')
ax7.axhline(30, color='#f0883e', linestyle='--', linewidth=1, label='30 dB — acceptable')
ax7.axhline(40, color='#3fb950', linestyle='--', linewidth=1, label='40 dB — excellent')
i_idx = [f for f,t in enumerate(frame_types) if t=='I']
ax7.plot(i_idx, psnr[i_idx], 'r^', markersize=7, label='I-frames')
ax7.set_xlabel('Frame index', color='#8b949e', fontsize=8)
ax7.set_ylabel('PSNR (dB)', color='#8b949e', fontsize=8)
ax7.set_title(
    f"STEP 7 — Decoding & Evaluation  |  "
    f"Mean PSNR = {overall_psnr:.1f} dB   |   "
    f"Compression Ratio = {overall_cr:.2f}×   |   "
    f"Compressed = {total_bits/8/1024:.1f} KB  /  Raw = {raw_bytes/1024:.1f} KB",
    **TITLE_STYLE)
ax7.legend(fontsize=8, facecolor='#161b22', labelcolor='white', loc='lower right')
ax7.tick_params(colors='#8b949e', labelsize=8)
ax7.grid(alpha=0.2)
for sp in ax7.spines.values(): sp.set_color('#30363d')

plt.subplots_adjust(left=0.06, right=0.98, top=0.98, bottom=0.03, hspace=0.55, wspace=0.35)
os.makedirs("outputs", exist_ok=True)
out_path = "outputs/compression_pipeline_all_steps.png"
plt.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='#0d1117')
print(f"✓ Saved: {out_path}")
plt.show()