"""
Video Compression Project - Complete Pipeline (Steps 1 to 7)
Suez Canal University - Faculty of Computers and Informatics
Computer Science Department

Python equivalent of the MATLAB implementation.
Requirements: pip install opencv-python numpy scipy matplotlib
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.fft import dctn, idctn
from collections import Counter
import heapq
import os
import sys

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def zigzag_scan(block):
    """Zig-zag scan of an 8x8 block into a 1x64 array."""
    order = np.array([
         1,  2,  6,  7, 15, 16, 28, 29,
         3,  5,  8, 14, 17, 27, 30, 43,
         4,  9, 13, 18, 26, 31, 42, 44,
        10, 12, 19, 25, 32, 41, 45, 54,
        11, 20, 24, 33, 40, 46, 53, 55,
        21, 23, 34, 39, 47, 52, 56, 61,
        22, 35, 38, 48, 51, 57, 60, 62,
        36, 37, 49, 50, 58, 59, 63, 64
    ]).reshape(8, 8) - 1  # 0-indexed

    flat = np.zeros(64)
    for r in range(8):
        for c in range(8):
            flat[order[r, c]] = block[r, c]
    return flat


def rle_encode(arr):
    """Run-Length Encoding: returns list of (value, count) pairs."""
    encoded = []
    count = 1
    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1]:
            count += 1
        else:
            encoded.append((arr[i - 1], count))
            count = 1
    encoded.append((arr[-1], count))
    return encoded


def rle_decode(encoded):
    """Decode RLE pairs back to flat array."""
    arr = []
    for val, count in encoded:
        arr.extend([val] * int(count))
    return np.array(arr)


def reconstruct_iframe(compressed_frame, Q, h, w, block_size=8):
    """Reconstruct Y channel from RLE+DCT compressed I-frame data."""
    Y_recon = np.zeros((h, w))
    num_bh = h // block_size
    num_bw = w // block_size
    block_num = 0

    for row in range(num_bh):
        for col in range(num_bw):
            if block_num >= len(compressed_frame):
                break
            encoded = compressed_frame[block_num]
            flat = rle_decode(encoded)
            if len(flat) < 64:
                flat = np.pad(flat, (0, 64 - len(flat)))
            q_block = flat[:64].reshape(8, 8)
            dct_block = q_block * Q
            block = idctn(dct_block, norm='ortho')
            r_start = row * block_size
            c_start = col * block_size
            Y_recon[r_start:r_start+8, c_start:c_start+8] = block
            block_num += 1

    return np.clip(Y_recon, 0, 255)


# ============================================================
# HUFFMAN CODING (No external library needed)
# ============================================================

class HuffmanNode:
    def __init__(self, symbol, prob):
        self.symbol = symbol
        self.prob   = prob
        self.left   = None
        self.right  = None

    def __lt__(self, other):
        return self.prob < other.prob


def build_huffman_dict(symbols, probs):
    """Build Huffman dictionary without any external library."""
    if len(symbols) == 1:
        return {symbols[0]: '0'}

    heap = [HuffmanNode(s, p) for s, p in zip(symbols, probs)]
    heapq.heapify(heap)

    while len(heap) > 1:
        left  = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.prob + right.prob)
        merged.left  = left
        merged.right = right
        heapq.heappush(heap, merged)

    root = heap[0]
    codes = {}

    def traverse(node, code):
        if node is None:
            return
        if node.symbol is not None:
            codes[node.symbol] = code if code else '0'
            return
        traverse(node.left,  code + '0')
        traverse(node.right, code + '1')

    traverse(root, '')
    return codes


def encode_with_dict(symbols, huff_dict):
    """Encode symbol array to binary string using Huffman dict."""
    return ''.join(huff_dict.get(s, '') for s in symbols)


def huffman_decode(encoded, huff_dict):
    """Fast Huffman decode using reverse lookup."""
    reverse = {v: k for k, v in huff_dict.items()}
    max_len  = max(len(c) for c in reverse)
    symbols  = []
    i = 0
    while i < len(encoded):
        matched = False
        for length in range(1, min(max_len, len(encoded) - i) + 1):
            codeword = encoded[i:i+length]
            if codeword in reverse:
                symbols.append(reverse[codeword])
                i += length
                matched = True
                break
        if not matched:
            break
    return np.array(symbols)


# ============================================================
# STEP 1: VIDEO INPUT HANDLING
# معالجة مدخلات الفيديو
# ============================================================
# EN: Read the video file frame by frame, then convert each frame
#     from BGR (OpenCV default) to YCbCr. The Y channel carries
#     brightness (luma) and is the main channel used for compression.
#
# AR: نقرأ ملف الفيديو إطاراً بإطار ونحوله من BGR إلى YCbCr.
#     قناة Y تحمل السطوع وهي القناة الرئيسية في الضغط.

print("==> STEP 1: Reading video and converting to YUV...")

VIDEO_PATH = "nature.mp4"

if not os.path.exists(VIDEO_PATH):
    # Generate synthetic video if file not found
    print(f"   '{VIDEO_PATH}' not found — generating synthetic video...")
    num_frames_gen = 30
    H_gen, W_gen   = 120, 160
    frames_rgb     = []
    for f in range(num_frames_gen):
        img = np.zeros((H_gen, W_gen, 3), dtype=np.uint8)
        offset = (f * 3) % (W_gen - 40)
        img[30:70, offset:offset+40] = [180, 100, 60]
        img += np.random.randint(0, 8, img.shape, dtype=np.uint8)
        frames_rgb.append(img)
    print(f"   Synthetic video: {num_frames_gen} frames of {H_gen}x{W_gen}")
else:
    cap = cv2.VideoCapture(VIDEO_PATH)
    frames_rgb = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"   Loaded '{VIDEO_PATH}'")

# Convert to YCbCr and extract Y channel
yuv_frames = []
for frame in frames_rgb:
    # OpenCV YCrCb conversion (equivalent to MATLAB rgb2ycbcr)
    yuv = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
    yuv_frames.append(yuv)

num_frames = len(frames_rgb)
frame_H    = frames_rgb[0].shape[0]
frame_W    = frames_rgb[0].shape[1]
print(f"   Total Frames: {num_frames}  |  Size: {frame_H}x{frame_W}")

# ---- Display: Step 1 ----
show_idx = np.round(np.linspace(0, num_frames-1, 4)).astype(int)
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
fig.suptitle("Step 1: Video Input - RGB vs Y channel | RGB مقابل قناة السطوع", fontsize=11)
for k, idx in enumerate(show_idx):
    axes[0, k].imshow(frames_rgb[idx])
    axes[0, k].set_title(f"RGB Frame {idx+1}", fontsize=8)
    axes[0, k].axis('off')

    axes[1, k].imshow(yuv_frames[idx][:,:,0], cmap='gray')
    axes[1, k].set_title(f"Y-channel {idx+1}", fontsize=8)
    axes[1, k].axis('off')
plt.tight_layout()
plt.savefig("step1_output.png", dpi=100, bbox_inches='tight')
plt.show()
print("   Figure saved: step1_output.png")


# ============================================================
# STEP 2: FRAME TYPE DECISION
# تحديد نوع كل إطار
# ============================================================
# EN: Every 10th frame (index 0,10,20,...) is an I-frame (Intra),
#     compressed independently. All others are P-frames (Predictive).
#
# AR: كل إطار عاشر I-frame يُضغط باستقلالية، والباقي P-frames.

print("\n==> STEP 2: Assigning frame types...")

frame_types = []
for f in range(num_frames):
    frame_types.append('I' if f % 10 == 0 else 'P')

num_I = frame_types.count('I')
num_P = frame_types.count('P')
print(f"   I-frames: {num_I}  |  P-frames: {num_P}")

# ---- Display: Step 2 ----
type_numeric = [1 if t == 'I' else 0 for t in frame_types]
fig, ax = plt.subplots(figsize=(12, 3))
ax.stem(range(num_frames), type_numeric, markerfmt='C0o', linefmt='C0-', basefmt='k-')
ax.set_yticks([0, 1]); ax.set_yticklabels(['P-frame', 'I-frame'])
ax.set_xlabel('Frame Index | رقم الإطار')
ax.set_title('Step 2: Frame Type Decision | نوع كل إطار  (I=1 , P=0)')
ax.grid(True)
plt.tight_layout()
plt.savefig("step2_output.png", dpi=100, bbox_inches='tight')
plt.show()
print("   Figure saved: step2_output.png")


# ============================================================
# STEP 3: INTRA-FRAME COMPRESSION (I-FRAMES)
# ضغط الإطارات المستقلة
# ============================================================
# EN: For each I-frame: extract Y channel, divide into 8x8 blocks,
#     apply DCT2, quantize, zig-zag scan, then RLE encode.
#
# AR: لكل I-frame: استخرج Y، قسّم إلى كتل 8x8، طبّق DCT2،
#     كمّم، مسح zig-zag، ثم RLE.

print("\n==> STEP 3: I-frame compression (DCT + Quantization + RLE)...")

Q              = np.ones((8, 8)) * 10.0
block_size     = 8
compressed_data     = [None] * num_frames
iframe_Y_quantized  = [None] * num_frames

for f in range(num_frames):
    if frame_types[f] == 'I':
        Y     = yuv_frames[f][:,:,0].astype(float)
        h, w  = Y.shape
        compressed_frame = []
        q_Y  = np.zeros((h, w))

        for x in range(0, h - block_size + 1, block_size):
            for y in range(0, w - block_size + 1, block_size):
                block     = Y[x:x+8, y:y+8]
                dct_block = dctn(block, norm='ortho')
                q_block   = np.round(dct_block / Q)

                flat    = zigzag_scan(q_block)
                encoded = rle_encode(flat)

                compressed_frame.append(encoded)
                q_Y[x:x+8, y:y+8] = q_block

        compressed_data[f]     = compressed_frame
        iframe_Y_quantized[f]  = q_Y
        print(f"   Frame {f+1:2d} [I] compressed into {len(compressed_frame)} blocks")

# ---- Display: Step 3 ----
first_I = next(f for f, t in enumerate(frame_types) if t == 'I')
Y_orig  = yuv_frames[first_I][:,:,0].astype(float)
Y_recon = reconstruct_iframe(compressed_data[first_I], Q,
                              Y_orig.shape[0], Y_orig.shape[1], block_size)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("Step 3: Intra-frame Compression (DCT + RLE) | ضغط الإطارات المستقلة", fontsize=11)

axes[0].imshow(Y_orig, cmap='gray', vmin=0, vmax=255)
axes[0].set_title(f"Original Y-channel (Frame {first_I+1}) | الأصلي")
axes[0].axis('off')

axes[1].imshow(iframe_Y_quantized[first_I], cmap='gray')
axes[1].set_title("Quantized DCT Coefficients | معاملات DCT المكمّمة")
axes[1].axis('off')

axes[2].imshow(Y_recon, cmap='gray', vmin=0, vmax=255)
axes[2].set_title("Reconstructed I-frame | الإطار المعاد بناؤه")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("step3_output.png", dpi=100, bbox_inches='tight')
plt.show()
print("   Figure saved: step3_output.png")


# ============================================================
# STEP 4: INTER-FRAME COMPRESSION (P-FRAMES)
# ضغط الإطارات التنبؤية
# ============================================================
# EN: P-frames store only the DIFFERENCE from the previous frame.
#     Block Matching finds motion vectors (dy,dx) for each 8x8 block.
#     residual = current block - predicted block.
#
# AR: الـ P-frames تخزّن فقط الفرق عن الإطار السابق.
#     مطابقة الكتل تجد متجهات الحركة لكل كتلة 8x8.
#     الباقي = الكتلة الحالية - الكتلة المتوقعة.

print("\n==> STEP 4: Inter-frame Compression (P-frames - Block Matching)...")

search_range   = 4
motion_vectors = [None] * num_frames
residuals      = [None] * num_frames

for f in range(num_frames):
    current_frame = yuv_frames[f][:,:,0].astype(float)
    h, w = current_frame.shape

    if frame_types[f] == 'I':
        motion_vectors[f] = None
        residuals[f] = reconstruct_iframe(compressed_data[f], Q, h, w, block_size)
        print(f"   Frame {f+1:2d} -> I-frame")

    else:
        ref_frame = yuv_frames[f-1][:,:,0].astype(float)
        num_bh = h // block_size
        num_bw = w // block_size
        mvs         = np.zeros((num_bh * num_bw, 2), dtype=int)
        residual_img = np.zeros((h, w))

        block_num = 0
        for row in range(num_bh):
            for col in range(num_bw):
                r_start = row * block_size
                c_start = col * block_size
                r_end   = r_start + block_size
                c_end   = c_start + block_size

                current_block = current_frame[r_start:r_end, c_start:c_end]

                best_mad = np.inf
                best_dy, best_dx = 0, 0

                for dy in range(-search_range, search_range + 1):
                    for dx in range(-search_range, search_range + 1):
                        r_ref = r_start + dy
                        c_ref = c_start + dx
                        if (r_ref < 0 or c_ref < 0 or
                                r_ref + block_size > h or c_ref + block_size > w):
                            continue
                        ref_block = ref_frame[r_ref:r_ref+block_size, c_ref:c_ref+block_size]
                        mad = np.mean(np.abs(current_block - ref_block))
                        if mad < best_mad:
                            best_mad = mad
                            best_dy, best_dx = dy, dx

                mvs[block_num] = [best_dy, best_dx]

                r_ref = np.clip(r_start + best_dy, 0, h - block_size)
                c_ref = np.clip(c_start + best_dx, 0, w - block_size)
                predicted_block = ref_frame[r_ref:r_ref+block_size, c_ref:c_ref+block_size]
                residual_img[r_start:r_end, c_start:c_end] = current_block - predicted_block

                block_num += 1

        motion_vectors[f] = mvs
        residuals[f]      = residual_img
        mean_mv = np.mean(np.sqrt(np.sum(mvs**2, axis=1)))
        print(f"   Frame {f+1:2d} -> P-frame | Mean MV: {mean_mv:.2f} px")

# ---- Display: Step 4 ----
first_P = next(f for f, t in enumerate(frame_types) if t == 'P')
h_f, w_f = residuals[first_P].shape
num_bh   = h_f // block_size
num_bw   = w_f // block_size

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Step 4: Inter-frame Compression | ضغط الإطارات التنبؤية", fontsize=11)

axes[0].imshow(yuv_frames[first_P][:,:,0], cmap='gray', vmin=0, vmax=255)
axes[0].set_title(f"Frame {first_P+1} - Original Y | الأصلي")
axes[0].axis('off')

im = axes[1].imshow(residuals[first_P], cmap='gray')
axes[1].set_title(f"Frame {first_P+1} - Residual | الباقي")
axes[1].axis('off')
plt.colorbar(im, ax=axes[1])

mvs_p = motion_vectors[first_P]
cols, rows = np.meshgrid(np.arange(1, num_bw+1), np.arange(1, num_bh+1))
mv_dy = mvs_p[:,0].reshape(num_bh, num_bw)
mv_dx = mvs_p[:,1].reshape(num_bh, num_bw)
axes[2].quiver(cols, rows, mv_dx, mv_dy, color='red', scale=30)
axes[2].set_xlim(0, num_bw+1); axes[2].set_ylim(0, num_bh+1)
axes[2].invert_yaxis()
axes[2].set_title(f"Motion Vectors (Frame {first_P+1}) | متجهات الحركة")
axes[2].set_xlabel("Block Col"); axes[2].set_ylabel("Block Row")

plt.tight_layout()
plt.savefig("step4_output.png", dpi=100, bbox_inches='tight')
plt.show()
print("   Figure saved: step4_output.png")


# ============================================================
# STEP 5: ENTROPY CODING (Manual Huffman)
# الترميز بالإنتروبيا - هافمان
# ============================================================
# EN: Huffman coding assigns shorter codes to frequent values and
#     longer codes to rare values. Applied to motion vectors and
#     residuals. Built manually with no external library.
#
# AR: هافمان يعطي رموزاً أقصر للقيم الأكثر تكراراً.
#     نبني الشجرة يدوياً بدون أي مكتبة خارجية.

print("\n==> STEP 5: Entropy Coding (Manual Huffman)...")

q_step            = 10
encoded_streams   = [None] * num_frames
huffman_dicts     = [None] * num_frames
original_size_bits = np.zeros(num_frames)
encoded_size_bits  = np.zeros(num_frames)

for f in range(num_frames):
    if frame_types[f] == 'I':
        quantized = np.round(residuals[f] / q_step).astype(int)
        symbols   = quantized.flatten()
    else:
        mv_flat   = motion_vectors[f].flatten()
        res_quant = np.round(residuals[f].flatten() / q_step).astype(int)
        symbols   = np.concatenate([mv_flat, res_quant])

    unique_syms, counts = np.unique(symbols, return_counts=True)
    probs = counts / counts.sum()

    huff_dict = build_huffman_dict(unique_syms.tolist(), probs.tolist())
    encoded   = encode_with_dict(symbols.tolist(), huff_dict)

    encoded_streams[f]      = encoded
    huffman_dicts[f]        = huff_dict
    original_size_bits[f]   = len(symbols) * 8
    encoded_size_bits[f]    = len(encoded)

    ratio = original_size_bits[f] / max(encoded_size_bits[f], 1)
    print(f"   Frame {f+1:2d} [{frame_types[f]}] | "
          f"Orig: {int(original_size_bits[f]):6d} bits | "
          f"Huffman: {int(encoded_size_bits[f]):6d} bits | "
          f"Ratio: {ratio:.2f}")

# ---- Display: Step 5 ----
x = np.arange(num_frames)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Step 5: Entropy Coding Results | نتائج الترميز بالإنتروبيا", fontsize=11)

axes[0].bar(x, original_size_bits, color=[0.2, 0.5, 0.8], label='Original Bits | الأصلي')
axes[0].bar(x, encoded_size_bits,  color=[0.9, 0.3, 0.3], label='Huffman Encoded | بعد هافمان')
axes[0].set_xlabel('Frame Number | رقم الإطار')
axes[0].set_ylabel('Bits')
axes[0].set_title('Bit Usage per Frame | حجم البيانات لكل إطار')
axes[0].legend(); axes[0].grid(True)

cr_per_frame = original_size_bits / np.maximum(encoded_size_bits, 1)
axes[1].plot(cr_per_frame, 'g-o', linewidth=1.5)
axes[1].set_xlabel('Frame Number | رقم الإطار')
axes[1].set_ylabel('Compression Ratio | نسبة الضغط')
axes[1].set_title('Huffman Ratio per Frame | نسبة ضغط هافمان لكل إطار')
axes[1].grid(True)

plt.tight_layout()
plt.savefig("step5_output.png", dpi=100, bbox_inches='tight')
plt.show()
print("   Figure saved: step5_output.png")


# ============================================================
# STEP 6: BITSTREAM FORMATION
# تكوين تدفق البيانات
# ============================================================
# EN: Package all encoded frames into a single bitstream.
#     Each frame has a header (index, type, size) like H.264.
#
# AR: نعبّئ جميع الإطارات في تدفق بيانات واحد مع رأس لكل إطار.

print("\n==> STEP 6: Bitstream Formation...")

bitstream = {
    'header': {
        'num_frames': num_frames,
        'frame_H':    frame_H,
        'frame_W':    frame_W,
        'block_size': block_size,
        'q_step':     q_step,
    },
    'frames': []
}

total_bitstream_bits = 0

for f in range(num_frames):
    header_bits = 72   # 32b index + 8b type + 32b size
    data_bits   = int(encoded_size_bits[f])
    frame_total = header_bits + data_bits

    bitstream['frames'].append({
        'index':        f,
        'type':         frame_types[f],
        'encoded_data': encoded_streams[f],
        'huff_dict':    huffman_dicts[f],
        'total_bits':   frame_total,
    })

    total_bitstream_bits += frame_total
    print(f"   Frame {f+1:2d} [{frame_types[f]}] packaged | {frame_total} bits")

raw_video_bytes = frame_H * frame_W * num_frames
print(f"\n   TOTAL Bitstream : {total_bitstream_bits/8/1024:.2f} KB")
print(f"   Raw Video Size  : {raw_video_bytes/1024:.2f} KB")
print(f"   Compression Ratio: {(raw_video_bytes*8)/total_bitstream_bits:.2f}x")

# ---- Display: Step 6 ----
frame_bits = [bitstream['frames'][f]['total_bits'] for f in range(num_frames)]
bar_colors = ['#D93333' if frame_types[f] == 'I' else '#3399D9' for f in range(num_frames)]

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Step 6: Bitstream Formation | تكوين تدفق البيانات", fontsize=11)

axes[0].bar(range(num_frames), frame_bits, color=bar_colors)
axes[0].set_xlabel('Frame Index | رقم الإطار')
axes[0].set_ylabel('Bits')
axes[0].set_title('Bits per Frame | حجم كل إطار في التدفق')
i_patch = mpatches.Patch(color='#D93333', label='I-frame')
p_patch = mpatches.Patch(color='#3399D9', label='P-frame')
axes[0].legend(handles=[i_patch, p_patch]); axes[0].grid(True)

axes[1].bar(['Raw Video | الخام', 'Compressed | المضغوط'],
            [raw_video_bytes * 8, total_bitstream_bits],
            color=[0.4, 0.7, 0.4])
axes[1].set_ylabel('Total Bits')
axes[1].set_title(f"Overall Compression: {(raw_video_bytes*8)/total_bitstream_bits:.2f}x | نسبة الضغط الكلية")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("step6_output.png", dpi=100, bbox_inches='tight')
plt.show()
print("   Figure saved: step6_output.png")


# ============================================================
# STEP 7: TESTING & EVALUATION (PSNR + Compression Ratio)
# الاختبار والتقييم
# ============================================================
# EN: Decode the bitstream back to video frames and compare with
#     originals using PSNR. PSNR > 30 dB = acceptable,  > 40 dB = excellent.
#
# AR: نفك ضغط التدفق ونقارن بالأصلي باستخدام PSNR.
#     فوق 30 ديسيبل مقبول، فوق 40 ممتاز.

print("\n==> STEP 7: Decoding and Evaluation (PSNR)...")

decoded_frames = np.zeros((frame_H, frame_W, num_frames))
psnr_values    = np.zeros(num_frames)

for f in range(num_frames):
    fd = bitstream['frames'][f]
    decoded_symbols = huffman_decode(fd['encoded_data'], fd['huff_dict'])

    h_f = yuv_frames[f][:,:,0].shape[0]
    w_f = yuv_frames[f][:,:,0].shape[1]
    num_bh = h_f // block_size
    num_bw = w_f // block_size

    if fd['type'] == 'I':
        num_pixels    = num_bh * num_bw * block_size * block_size
        decoded_quant = decoded_symbols[:num_pixels]
        reconstructed = decoded_quant.astype(float) * q_step
        reconstructed = reconstructed.reshape(h_f, w_f)
        reconstructed = np.clip(reconstructed, 0, 255)
        decoded_frames[:,:,f] = reconstructed[:frame_H, :frame_W]

    else:
        num_blocks   = num_bh * num_bw
        num_mv_syms  = num_blocks * 2
        num_pixels   = num_bh * num_bw * block_size * block_size

        if len(decoded_symbols) < num_mv_syms + num_pixels:
            decoded_symbols = np.pad(decoded_symbols,
                                     (0, num_mv_syms + num_pixels - len(decoded_symbols)))

        mv_decoded   = decoded_symbols[:num_mv_syms].reshape(num_blocks, 2)
        res_decoded  = decoded_symbols[num_mv_syms:num_mv_syms+num_pixels]
        residual_rec = res_decoded.astype(float) * q_step
        residual_rec = residual_rec.reshape(h_f, w_f)

        ref_decoded = decoded_frames[:,:,f-1]
        predicted   = np.zeros((h_f, w_f))

        block_num = 0
        for row in range(num_bh):
            for col in range(num_bw):
                r_start = row * block_size
                c_start = col * block_size
                dy = int(mv_decoded[block_num, 0])
                dx = int(mv_decoded[block_num, 1])
                r_ref = np.clip(r_start + dy, 0, h_f - block_size)
                c_ref = np.clip(c_start + dx, 0, w_f - block_size)
                predicted[r_start:r_start+block_size, c_start:c_start+block_size] = \
                    ref_decoded[r_ref:r_ref+block_size, c_ref:c_ref+block_size]
                block_num += 1

        reconstructed = np.clip(predicted + residual_rec, 0, 255)
        decoded_frames[:,:,f] = reconstructed[:frame_H, :frame_W]

    # PSNR
    orig  = yuv_frames[f][:,:,0].astype(float)
    recon = decoded_frames[:,:,f]
    mse   = np.mean((orig - recon) ** 2)
    psnr_values[f] = 10 * np.log10(255**2 / max(mse, 1e-10))
    print(f"   Frame {f+1:2d} [{fd['type']}] | PSNR: {psnr_values[f]:.2f} dB")

# ---- Final Summary ----
overall_psnr = np.mean(psnr_values)
overall_cr   = (raw_video_bytes * 8) / total_bitstream_bits

print("\n==========================================")
print("  FINAL EVALUATION SUMMARY")
print("==========================================")
print(f"  Average PSNR       : {overall_psnr:.2f} dB")
print(f"  Compression Ratio  : {overall_cr:.2f}x")
print(f"  Raw Video Size     : {raw_video_bytes/1024:.2f} KB")
print(f"  Compressed Size    : {total_bitstream_bits/8/1024:.2f} KB")
print("==========================================")

# ---- Display: Step 7 - Original vs Decoded ----
show_frames = list(dict.fromkeys([0, 1,
                                   next(f for f,t in enumerate(frame_types) if t=='I')+1,
                                   num_frames-1]))
show_frames = [f for f in show_frames if f < num_frames][:4]

fig, axes = plt.subplots(2, len(show_frames), figsize=(4*len(show_frames), 6))
fig.suptitle("Step 7: Original vs Decoded | الأصلي مقابل المُستعاد", fontsize=11)

for k, f in enumerate(show_frames):
    axes[0, k].imshow(yuv_frames[f][:,:,0], cmap='gray', vmin=0, vmax=255)
    axes[0, k].set_title(f"Orig F{f+1} [{frame_types[f]}]", fontsize=8)
    axes[0, k].axis('off')

    axes[1, k].imshow(decoded_frames[:,:,f], cmap='gray', vmin=0, vmax=255)
    axes[1, k].set_title(f"Decoded F{f+1}\nPSNR={psnr_values[f]:.1f}dB", fontsize=8)
    axes[1, k].axis('off')

plt.tight_layout()
plt.savefig("step7_comparison.png", dpi=100, bbox_inches='tight')
plt.show()

# ---- Display: Step 7 - PSNR & Compression Pie ----
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Step 7: Evaluation - PSNR & Compression | التقييم النهائي", fontsize=11)

axes[0].plot(psnr_values, 'b-o', linewidth=1.5, markersize=4, label='PSNR')
axes[0].axhline(30, color='r', linestyle='--', label='30 dB (Acceptable)')
axes[0].axhline(40, color='g', linestyle='--', label='40 dB (Excellent)')
i_indices = [f for f, t in enumerate(frame_types) if t == 'I']
axes[0].plot(i_indices, psnr_values[i_indices], 'r^', markersize=8, label='I-frames')
axes[0].set_xlabel('Frame Number | رقم الإطار')
axes[0].set_ylabel('PSNR (dB)')
axes[0].set_title(f"PSNR per Frame | Mean: {overall_psnr:.2f} dB")
axes[0].legend(loc='lower right'); axes[0].grid(True)

compressed_kb = total_bitstream_bits / 8
saved_kb      = max(raw_video_bytes - compressed_kb, 0)
axes[1].pie(
    [compressed_kb, saved_kb],
    labels=[f"Compressed\n{compressed_kb/1024:.1f} KB",
            f"Saved\n{saved_kb/1024:.1f} KB"],
    colors=['#E84C4C', '#4CAF50'],
    autopct='%1.1f%%'
)
axes[1].set_title(f"Compression Ratio: {overall_cr:.2f}x")

plt.tight_layout()
plt.savefig("step7_psnr.png", dpi=100, bbox_inches='tight')
plt.show()
print("   Figures saved: step7_comparison.png, step7_psnr.png")

print("\n==> ALL STEPS COMPLETE! All figures are shown.")