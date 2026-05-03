%% Video Compression Project - Complete Pipeline (Steps 1 to 7)
% Suez Canal University - Faculty of Computers and Informatics
% Computer Science Department

clc; clear; close all;

%% ============================================================
%% STEP 1: VIDEO INPUT HANDLING
%% معالجة مدخلات الفيديو
%% ============================================================
% EN: Read the video file frame by frame, then convert each frame
%     from RGB color space to YUV (YCbCr). The Y channel carries
%     brightness (luma) and is the main channel used for compression.
%
% AR: نقرأ ملف الفيديو إطاراً بإطار، ثم نحوّل كل إطار من نظام
%     الألوان RGB إلى YUV. قناة Y تحمل السطوع وهي القناة الرئيسية
%     المستخدمة في الضغط.

fprintf('==> STEP 1: Reading video and converting to YUV...\n');

video = VideoReader('./nature.mp4');

frames     = {};
yuv_frames = {};
i = 1;

while hasFrame(video)
    frame        = readFrame(video);
    frames{i}    = frame;
    yuv_frames{i} = rgb2ycbcr(frame);
    i = i + 1;
end

numFrames = length(frames);
frameH    = size(frames{1}, 1);
frameW    = size(frames{1}, 2);
fprintf('   Total Frames: %d  |  Size: %dx%d\n', numFrames, frameH, frameW);

%% ---- Display: Step 1 output ----
figure('Name','STEP 1 - Original vs YUV Frames','NumberTitle','off');
showIdx = round(linspace(1, numFrames, 4));
for k = 1:4
    subplot(2,4,k);
    imshow(frames{showIdx(k)});
    title(sprintf('RGB Frame %d', showIdx(k)), 'FontSize', 8);

    subplot(2,4,k+4);
    imshow(yuv_frames{showIdx(k)}(:,:,1), []);   % Y channel only
    title(sprintf('Y-channel %d', showIdx(k)), 'FontSize', 8);
end
sgtitle('Step 1: Video Input - RGB vs Y channel | RGB مقابل قناة السطوع');


%% ============================================================
%% STEP 2: FRAME TYPE DECISION
%% تحديد نوع كل إطار
%% ============================================================
% EN: Every 10th frame (index 1,11,21,...) is an I-frame (Intra),
%     which is compressed independently. All other frames are
%     P-frames (Predictive), which store only the difference from
%     the previous frame.
%
% AR: كل إطار عاشر يُعدّ I-frame يُضغط باستقلالية تامة.
%     باقي الإطارات P-frames تخزّن فقط الفرق عن الإطار السابق.

fprintf('\n==> STEP 2: Assigning frame types...\n');

frame_types = strings(1, numFrames);
for f = 1:numFrames
    if mod(f, 10) == 1
        frame_types(f) = "I";
    else
        frame_types(f) = "P";
    end
end

numI = sum(frame_types == "I");
numP = sum(frame_types == "P");
fprintf('   I-frames: %d  |  P-frames: %d\n', numI, numP);

%% ---- Display: Step 2 output ----
figure('Name','STEP 2 - Frame Type Map','NumberTitle','off');
typeNumeric = double(frame_types == "I");   % 1=I, 0=P
stem(typeNumeric, 'filled', 'MarkerSize', 5);
yticks([0 1]); yticklabels({'P-frame','I-frame'});
xlabel('Frame Index | رقم الإطار');
title('Step 2: Frame Type Decision | نوع كل إطار  (I=1 , P=0)');
grid on;


%% ============================================================
%% STEP 3: INTRA-FRAME COMPRESSION (I-FRAMES)
%% ضغط الإطارات المستقلة
%% ============================================================
% EN: For each I-frame we:
%   1. Extract the Y (luma) channel
%   2. Divide it into 8x8 pixel blocks
%   3. Apply DCT2 to each block (frequency transform)
%   4. Quantize DCT coefficients by dividing by Q matrix
%   5. Flatten using zig-zag order and apply Run-Length Encoding (RLE)
%
% AR: لكل I-frame نقوم بـ:
%   1. استخراج قناة Y (السطوع)
%   2. تقسيمها إلى كتل 8x8 بكسل
%   3. تطبيق تحويل DCT2 على كل كتلة
%   4. تكميم معاملات DCT بالقسمة على مصفوفة Q
%   5. تسطيح المصفوفة بترتيب zig-zag وتطبيق RLE

fprintf('\n==> STEP 3: I-frame compression (DCT + Quantization + RLE)...\n');

Q               = ones(8,8) * 10;    % quantization matrix
compressed_data = cell(1, numFrames);
blockSize       = 8;

% also store full quantized Y for later use in Step 4
iframeY_quantized = cell(1, numFrames);

for f = 1:numFrames
    if frame_types(f) == "I"
        yuv   = yuv_frames{f};
        Y     = double(yuv(:,:,1));
        [h,w] = size(Y);

        compressed_frame = {};
        q_Y = zeros(h, w);    % store full quantized image
        idx  = 1;

        for x = 1:blockSize:h-blockSize+1
            for y = 1:blockSize:w-blockSize+1
                block     = Y(x:x+7, y:y+7);
                dct_block = dct2(block);
                q_block   = round(dct_block ./ Q);

                % Zig-zag flatten then RLE
                flat    = zigzag_scan(q_block);
                encoded = rle_encode(flat);

                compressed_frame{idx} = encoded;
                q_Y(x:x+7, y:y+7)    = q_block;
                idx = idx + 1;
            end
        end

        compressed_data{f}      = compressed_frame;
        iframeY_quantized{f}    = q_Y;
        fprintf('   Frame %2d [I] compressed into %d blocks\n', f, idx-1);
    end
end

%% ---- Display: Step 3 output ----
% Find first I-frame
firstI = find(frame_types == "I", 1);
yuv_tmp = yuv_frames{firstI};
Y_orig  = double(yuv_tmp(:,:,1));

figure('Name','STEP 3 - I-frame DCT Compression','NumberTitle','off');

subplot(1,3,1);
imshow(uint8(Y_orig), []);
title(sprintf('Original Y-channel (Frame %d) | الأصلي', firstI));

subplot(1,3,2);
imshow(iframeY_quantized{firstI}, []);
title('Quantized DCT Coefficients | معاملات DCT المكمّمة');

% Reconstruct first I-frame to show visually
Y_recon = reconstructIframe(compressed_data{firstI}, Q, size(Y_orig,1), size(Y_orig,2), blockSize);
subplot(1,3,3);
imshow(uint8(Y_recon), []);
title('Reconstructed I-frame | الإطار المعاد بناؤه');

sgtitle('Step 3: Intra-frame Compression (DCT + RLE) | ضغط الإطارات المستقلة');


%% ============================================================
%% STEP 4: INTER-FRAME COMPRESSION (P-FRAMES)
%% ضغط الإطارات التنبؤية
%% ============================================================
% EN: P-frames store only the DIFFERENCE from the previous frame.
%     We use Block Matching to find motion vectors (dy,dx) for each
%     8x8 block. The residual = current block - predicted block.
%     This saves space since consecutive frames are very similar.
%
% AR: الـ P-frames لا تخزّن الصورة الكاملة بل فقط الفرق عن الإطار
%     السابق. نستخدم مطابقة الكتل لإيجاد متجهات الحركة (dy,dx)
%     لكل كتلة 8x8. الباقي = الكتلة الحالية - الكتلة المتوقعة.
%     هذا يوفر مساحة لأن الإطارات المتتالية متشابهة جداً.

fprintf('\n==> STEP 4: Inter-frame Compression (P-frames - Block Matching)...\n');

searchRange   = 4;
motionVectors = cell(1, numFrames);
residuals     = cell(1, numFrames);
frameTypes    = cell(1, numFrames);

for f = 1:numFrames
    yuv          = yuv_frames{f};
    currentFrame = double(yuv(:,:,1));   % use Y channel
    [h, w]       = size(currentFrame);

    if frame_types(f) == "I"
        frameTypes{f}    = 'I';
        motionVectors{f} = [];
        % Use reconstructed I-frame as residual base
        residuals{f} = reconstructIframe(compressed_data{f}, Q, h, w, blockSize);
        fprintf('   Frame %2d -> I-frame\n', f);

    else
        frameTypes{f} = 'P';
        refYUV   = yuv_frames{f-1};
        refFrame = double(refYUV(:,:,1));

        numBlocksH = floor(h / blockSize);
        numBlocksW = floor(w / blockSize);
        mvs        = zeros(numBlocksH * numBlocksW, 2);
        residualImg = zeros(h, w);

        blockNum = 0;
        for row = 1:numBlocksH
            for col = 1:numBlocksW
                blockNum = blockNum + 1;

                rStart = (row-1)*blockSize + 1;
                cStart = (col-1)*blockSize + 1;
                rEnd   = rStart + blockSize - 1;
                cEnd   = cStart + blockSize - 1;

                currentBlock = currentFrame(rStart:rEnd, cStart:cEnd);

                bestMAD = inf;
                bestDy  = 0; bestDx = 0;

                for dy = -searchRange:searchRange
                    for dx = -searchRange:searchRange
                        rRef = rStart + dy;
                        cRef = cStart + dx;
                        if rRef < 1 || cRef < 1 || ...
                           rRef+blockSize-1 > h || cRef+blockSize-1 > w
                            continue;
                        end
                        refBlock = refFrame(rRef:rRef+blockSize-1, cRef:cRef+blockSize-1);
                        MAD = mean(abs(currentBlock(:) - refBlock(:)));
                        if MAD < bestMAD
                            bestMAD = MAD;
                            bestDy  = dy;
                            bestDx  = dx;
                        end
                    end
                end

                mvs(blockNum, :) = [bestDy, bestDx];

                rRef = rStart + bestDy;
                cRef = cStart + bestDx;
                rRef = max(1, min(h-blockSize+1, rRef));
                cRef = max(1, min(w-blockSize+1, cRef));
                predictedBlock = refFrame(rRef:rRef+blockSize-1, cRef:cRef+blockSize-1);
                residualImg(rStart:rEnd, cStart:cEnd) = currentBlock - predictedBlock;
            end
        end

        motionVectors{f} = mvs;
        residuals{f}     = residualImg;
        fprintf('   Frame %2d -> P-frame | Mean MV: %.2f px\n', f, mean(sqrt(sum(mvs.^2,2))));
    end
end

%% ---- Display: Step 4 output ----
% Pick first P-frame
firstP = find(frame_types == "P", 1);

figure('Name','STEP 4 - Inter-frame Compression','NumberTitle','off');

subplot(1,3,1);
imshow(uint8(double(yuv_frames{firstP}(:,:,1))), []);
title(sprintf('Frame %d - Original Y | الأصلي', firstP));

subplot(1,3,2);
imagesc(residuals{firstP}); colormap(gca, gray); colorbar; axis image;
title(sprintf('Frame %d - Residual | الباقي', firstP));

subplot(1,3,3);
mvs_p        = motionVectors{firstP};
[h_f, w_f]   = size(residuals{firstP});
numBH        = floor(h_f/blockSize);
numBW        = floor(w_f/blockSize);
[bCols,bRows]= meshgrid(1:numBW, 1:numBH);
mvDy = reshape(mvs_p(:,1), numBH, numBW);
mvDx = reshape(mvs_p(:,2), numBH, numBW);
quiver(bCols, bRows, mvDx, mvDy, 'r');
axis([0 numBW+1 0 numBH+1]);
set(gca,'YDir','reverse');
title(sprintf('Motion Vectors (Frame %d) | متجهات الحركة', firstP));
xlabel('Block Col'); ylabel('Block Row');

sgtitle('Step 4: Inter-frame Compression | ضغط الإطارات التنبؤية');


%% ============================================================
%% STEP 5: ENTROPY CODING (Manual Huffman - No Toolbox)
%% الترميز بالإنتروبيا - هافمان
%% ============================================================
% EN: Huffman coding assigns shorter binary codes to frequent values
%     and longer codes to rare values. Applied to motion vectors and
%     residuals. The result is a compact binary string per frame.
%     No toolbox is needed - we build the Huffman tree manually.
%
% AR: ترميز هافمان يعطي رموزاً ثنائية أقصر للقيم الأكثر تكراراً
%     وأطول للنادرة. نطبقه على متجهات الحركة والباقيات.
%     لا نحتاج أي Toolbox - نبني شجرة هافمان يدوياً.

fprintf('\n==> STEP 5: Entropy Coding (Manual Huffman)...\n');

qStep            = 10;
encodedStream    = cell(1, numFrames);
huffmanDicts     = cell(1, numFrames);
originalSizeBits = zeros(1, numFrames);
encodedSizeBits  = zeros(1, numFrames);

for f = 1:numFrames
    if strcmp(frameTypes{f}, 'I')
        quantizedFrame = round(residuals{f} / qStep);
        symbols = int32(quantizedFrame(:)');
    else
        mvFlat   = int32(motionVectors{f}(:)');
        resQuant = int32(round(residuals{f}(:)' / qStep));
        symbols  = [mvFlat, resQuant];
    end

    uniqueSymbols = unique(double(symbols));
    counts = zeros(1, numel(uniqueSymbols));
    for ii = 1:numel(uniqueSymbols)
        counts(ii) = sum(double(symbols) == uniqueSymbols(ii));
    end
    probs = counts / sum(counts);

    dict    = buildHuffmanDict(uniqueSymbols, probs);
    encoded = encodeWithDict(double(symbols), uniqueSymbols, dict);

    encodedStream{f}    = encoded;
    huffmanDicts{f}     = dict;
    originalSizeBits(f) = numel(symbols) * 8;
    encodedSizeBits(f)  = numel(encoded);

    fprintf('   Frame %2d [%s] | Orig: %6d bits | Huffman: %6d bits | Ratio: %.2f\n', ...
        f, frameTypes{f}, originalSizeBits(f), encodedSizeBits(f), ...
        originalSizeBits(f)/max(encodedSizeBits(f),1));
end

%% ---- Display: Step 5 output ----
figure('Name','STEP 5 - Entropy Coding','NumberTitle','off');

subplot(1,2,1);
bar(originalSizeBits, 'FaceColor',[0.2 0.5 0.8]); hold on;
bar(encodedSizeBits,  'FaceColor',[0.9 0.3 0.3]);
legend('Original Bits | الأصلي','Huffman Encoded | بعد هافمان','Location','northeast');
xlabel('Frame Number | رقم الإطار'); ylabel('Bits');
title('Bit Usage per Frame | حجم البيانات لكل إطار'); grid on;

subplot(1,2,2);
crPerFrame = originalSizeBits ./ max(encodedSizeBits, 1);
plot(crPerFrame, 'g-o', 'LineWidth', 1.5);
xlabel('Frame Number | رقم الإطار');
ylabel('Compression Ratio | نسبة الضغط');
title('Huffman Ratio per Frame | نسبة ضغط هافمان لكل إطار'); grid on;

sgtitle('Step 5: Entropy Coding Results | نتائج الترميز بالإنتروبيا');


%% ============================================================
%% STEP 6: BITSTREAM FORMATION
%% تكوين تدفق البيانات
%% ============================================================
% EN: All encoded frames are packaged into a single bitstream.
%     Each frame has a small header (frame index, type, data size).
%     This mirrors how real formats like H.264 store video data.
%
% AR: جميع الإطارات المشفرة تُعبّأ في تدفق بيانات واحد.
%     كل إطار له رأس صغير يحتوي على رقمه ونوعه وحجم بياناته.
%     هذا يشبه كيفية عمل صيغ الفيديو الحقيقية مثل H.264.

fprintf('\n==> STEP 6: Bitstream Formation...\n');

bitstream.header.numFrames = numFrames;
bitstream.header.frameH    = frameH;
bitstream.header.frameW    = frameW;
bitstream.header.blockSize = blockSize;
bitstream.header.qStep     = qStep;
bitstream.frames           = struct();

totalBitstreamBits = 0;

for f = 1:numFrames
    headerBits = 72;   % 32b index + 8b type + 32b size
    dataBits   = encodedSizeBits(f);

    bitstream.frames(f).index       = f;
    bitstream.frames(f).type        = frameTypes{f};
    bitstream.frames(f).encodedData = encodedStream{f};
    bitstream.frames(f).huffDict    = huffmanDicts{f};
    bitstream.frames(f).totalBits   = headerBits + dataBits;

    totalBitstreamBits = totalBitstreamBits + headerBits + dataBits;
    fprintf('   Frame %2d [%s] packaged | %d bits\n', f, frameTypes{f}, bitstream.frames(f).totalBits);
end

rawVideoBytes = frameH * frameW * numFrames;
fprintf('   TOTAL Bitstream : %.2f KB\n', totalBitstreamBits/8/1024);
fprintf('   Raw Video Size  : %.2f KB\n', rawVideoBytes/1024);
fprintf('   Compression Ratio: %.2fx\n', (rawVideoBytes*8)/totalBitstreamBits);

%% ---- Display: Step 6 output ----
figure('Name','STEP 6 - Bitstream Formation','NumberTitle','off');

frameBits  = arrayfun(@(f) bitstream.frames(f).totalBits, 1:numFrames);
barColors  = zeros(numFrames, 3);
for f = 1:numFrames
    if strcmp(bitstream.frames(f).type,'I')
        barColors(f,:) = [0.85 0.2 0.2];
    else
        barColors(f,:) = [0.2 0.6 0.85];
    end
end

subplot(1,2,1);
b = bar(frameBits,'FaceColor','flat');
b.CData = barColors;
xlabel('Frame Index | رقم الإطار'); ylabel('Bits');
title('Bits per Frame | حجم كل إطار في التدفق');
legend([patch(NaN,NaN,[0.85 0.2 0.2]), patch(NaN,NaN,[0.2 0.6 0.85])], ...
    'I-frame','P-frame','Location','northeast');
grid on;

subplot(1,2,2);
bar([rawVideoBytes*8, totalBitstreamBits], 'FaceColor',[0.4 0.7 0.4]);
set(gca,'XTickLabel',{'Raw Video | الخام','Compressed | المضغوط'});
ylabel('Total Bits');
title(sprintf('Overall Compression: %.2fx | نسبة الضغط الكلية', (rawVideoBytes*8)/totalBitstreamBits));
grid on;

sgtitle('Step 6: Bitstream Formation | تكوين تدفق البيانات');


%% ============================================================
%% STEP 7: TESTING & EVALUATION (PSNR + Compression Ratio)
%% الاختبار والتقييم
%% ============================================================
% EN: We decode the bitstream back to video and compare with the
%     original using PSNR (Peak Signal-to-Noise Ratio).
%     PSNR > 30 dB = acceptable quality
%     PSNR > 40 dB = excellent quality
%     We also report the final compression ratio.
%
% AR: نفك ضغط التدفق ونقارن بالأصلي باستخدام PSNR.
%     PSNR فوق 30 ديسيبل = جودة مقبولة
%     PSNR فوق 40 ديسيبل = جودة ممتازة
%     كما نُبلّغ عن نسبة الضغط النهائية.

fprintf('\n==> STEP 7: Decoding and Evaluation (PSNR)...\n');

decodedFrames = zeros(frameH, frameW, numFrames, 'double');
psnrValues    = zeros(1, numFrames);

for f = 1:numFrames
    frameData = bitstream.frames(f);

    % Fast Huffman decode
    decodedSymbols = manualHuffmanDecode(frameData.encodedData, frameData.huffDict);

    [h_f, w_f] = size(double(yuv_frames{f}(:,:,1)));
    numBH = floor(h_f / blockSize);
    numBW = floor(w_f / blockSize);

    if strcmp(frameData.type, 'I')
        numPixels    = numBH * numBW * blockSize * blockSize;
        decodedQuant = decodedSymbols(1:numPixels);
        reconstructed = reshape(double(decodedQuant) * qStep, h_f, w_f);
        reconstructed = max(0, min(255, reconstructed));
        decodedFrames(:,:,f) = reconstructed(1:frameH, 1:frameW);

    else
        numBlocks    = numBH * numBW;
        numMVsyms    = numBlocks * 2;
        numPixels    = numBH * numBW * blockSize * blockSize;

        if numel(decodedSymbols) < numMVsyms + numPixels
            decodedSymbols(end+1 : numMVsyms+numPixels) = 0;
        end

        mvDecoded   = double(decodedSymbols(1:numMVsyms));
        resDecoded  = double(decodedSymbols(numMVsyms+1 : numMVsyms+numPixels));

        mvMatrix    = reshape(mvDecoded, numBlocks, 2);
        residualRec = reshape(resDecoded * qStep, h_f, w_f);

        refDecoded  = decodedFrames(:,:,f-1);
        predicted   = zeros(h_f, w_f);

        blockNum = 0;
        for row = 1:numBH
            for col = 1:numBW
                blockNum = blockNum + 1;
                rStart = (row-1)*blockSize + 1;
                cStart = (col-1)*blockSize + 1;
                rEnd   = rStart + blockSize - 1;
                cEnd   = cStart + blockSize - 1;

                dy   = mvMatrix(blockNum,1);
                dx   = mvMatrix(blockNum,2);
                rRef = max(1, min(h_f-blockSize+1, rStart+dy));
                cRef = max(1, min(w_f-blockSize+1, cStart+dx));

                predicted(rStart:rEnd, cStart:cEnd) = ...
                    refDecoded(rRef:rRef+blockSize-1, cRef:cRef+blockSize-1);
            end
        end

        reconstructed = predicted + residualRec;
        reconstructed = max(0, min(255, reconstructed));
        decodedFrames(:,:,f) = reconstructed(1:frameH, 1:frameW);
    end

    % PSNR
    orig  = double(yuv_frames{f}(:,:,1));
    recon = decodedFrames(:,:,f);
    mse   = mean((orig(:) - recon(:)).^2);
    psnrValues(f) = 10 * log10(255^2 / max(mse, 1e-10));

    fprintf('   Frame %2d [%s] | PSNR: %.2f dB\n', f, frameData.type, psnrValues(f));
end

%% ---- Final Summary ----
overallPSNR = mean(psnrValues);
overallCR   = (rawVideoBytes*8) / totalBitstreamBits;

fprintf('\n==========================================\n');
fprintf('  FINAL EVALUATION SUMMARY\n');
fprintf('==========================================\n');
fprintf('  Average PSNR       : %.2f dB\n', overallPSNR);
fprintf('  Compression Ratio  : %.2fx\n',   overallCR);
fprintf('  Raw Video Size     : %.2f KB\n', rawVideoBytes/1024);
fprintf('  Compressed Size    : %.2f KB\n', totalBitstreamBits/8/1024);
fprintf('==========================================\n');

%% ---- Display: Step 7 - Original vs Decoded ----
figure('Name','STEP 7 - Original vs Decoded Frames','NumberTitle','off');
showFrames = unique([1, 2, find(frame_types=="I",1)+1, numFrames]);
showFrames = showFrames(showFrames <= numFrames);
showFrames = showFrames(1:min(4,end));

for k = 1:numel(showFrames)
    f = showFrames(k);
    subplot(2, numel(showFrames), k);
    imshow(uint8(double(yuv_frames{f}(:,:,1))), []);
    title(sprintf('Orig F%d [%s]', f, frameTypes{f}), 'FontSize',8);

    subplot(2, numel(showFrames), k + numel(showFrames));
    imshow(uint8(decodedFrames(:,:,f)), []);
    title(sprintf('Decoded F%d\nPSNR=%.1fdB', f, psnrValues(f)), 'FontSize',8);
end
sgtitle('Step 7: Original vs Decoded | الأصلي مقابل المُستعاد');

%% ---- Display: Step 7 - PSNR & Compression ----
figure('Name','STEP 7 - PSNR and Compression Ratio','NumberTitle','off');

subplot(1,2,1);
plot(psnrValues, 'b-o', 'LineWidth',1.5,'MarkerSize',4); hold on;
yline(30,'r--','30 dB (Acceptable)','LabelHorizontalAlignment','left');
yline(40,'g--','40 dB (Excellent)', 'LabelHorizontalAlignment','left');
iIdx = find(strcmp(frameTypes,'I'));
plot(iIdx, psnrValues(iIdx), 'r^', 'MarkerSize',8, 'DisplayName','I-frames');
xlabel('Frame Number | رقم الإطار'); ylabel('PSNR (dB)');
title(sprintf('PSNR per Frame | Mean: %.2f dB', overallPSNR));
legend('PSNR','30dB','40dB','I-frames','Location','southeast'); grid on;

subplot(1,2,2);
pie([totalBitstreamBits/8, max(rawVideoBytes - totalBitstreamBits/8, 0)], ...
    {sprintf('Compressed\n%.1f KB', totalBitstreamBits/8/1024), ...
     sprintf('Saved\n%.1f KB', max(rawVideoBytes - totalBitstreamBits/8,0)/1024)});
title(sprintf('Compression Ratio: %.2fx', overallCR));
colormap([0.9 0.3 0.3; 0.3 0.7 0.3]);

sgtitle('Step 7: Evaluation - PSNR & Compression | التقييم النهائي');

fprintf('\n==> ALL STEPS COMPLETE! All figures are shown.\n');


%% ============================================================
%% LOCAL HELPER FUNCTIONS
%% ============================================================

function flat = zigzag_scan(block)
% Zig-zag scan of an 8x8 block into a 1x64 vector
    order = [
         1  2  6  7 15 16 28 29
         3  5  8 14 17 27 30 43
         4  9 13 18 26 31 42 44
        10 12 19 25 32 41 45 54
        11 20 24 33 40 46 53 55
        21 23 34 39 47 52 56 61
        22 35 38 48 51 57 60 62
        36 37 49 50 58 59 63 64
    ];
    flat = zeros(1,64);
    for r = 1:8
        for c = 1:8
            flat(order(r,c)) = block(r,c);
        end
    end
end

function encoded = rle_encode(arr)
% Run-Length Encoding: returns [value, count] pairs
    encoded = [];
    count   = 1;
    for i = 2:length(arr)
        if arr(i) == arr(i-1)
            count = count + 1;
        else
            encoded = [encoded; arr(i-1), count]; %#ok<AGROW>
            count = 1;
        end
    end
    encoded = [encoded; arr(end), count];
end

function Y_recon = reconstructIframe(compressedFrame, Q, h, w, blockSize)
% Reconstruct Y channel from RLE+DCT compressed I-frame data
    Y_recon  = zeros(h, w);
    numBH    = floor(h / blockSize);
    numBW    = floor(w / blockSize);
    blockNum = 0;
    for row = 1:numBH
        for col = 1:numBW
            blockNum = blockNum + 1;
            if blockNum > numel(compressedFrame); break; end
            encoded  = compressedFrame{blockNum};
            flat     = rle_decode(encoded);
            if numel(flat) < 64; flat(end+1:64) = 0; end
            q_block  = reshape(flat(1:64), 8, 8);
            dct_block= q_block .* Q;
            block    = idct2(dct_block);
            rStart   = (row-1)*blockSize + 1;
            cStart   = (col-1)*blockSize + 1;
            Y_recon(rStart:rStart+7, cStart:cStart+7) = block;
        end
    end
    Y_recon = max(0, min(255, Y_recon));
end

function arr = rle_decode(encoded)
% Decode RLE pairs back to flat array
    arr = [];
    for i = 1:size(encoded,1)
        arr = [arr, repmat(encoded(i,1), 1, encoded(i,2))]; %#ok<AGROW>
    end
end

function dict = buildHuffmanDict(symbols, probs)
% Build Huffman dictionary (no toolbox required)
    n = numel(probs);
    if n == 1
        dict = {symbols(1), '0'};
        return;
    end

    treeLeft  = zeros(1, 2*n);
    treeRight = zeros(1, 2*n);
    treeProb  = zeros(1, 2*n);
    treeProb(1:n) = probs;
    nextNode    = n + 1;
    activeNodes = 1:n;
    activeProbs = probs;

    while numel(activeNodes) > 1
        [~, sortIdx] = sort(activeProbs);
        i1 = sortIdx(1); i2 = sortIdx(2);
        treeLeft(nextNode)  = activeNodes(i1);
        treeRight(nextNode) = activeNodes(i2);
        treeProb(nextNode)  = activeProbs(i1) + activeProbs(i2);
        keep = setdiff(1:numel(activeNodes), [i1 i2]);
        activeNodes = [activeNodes(keep), nextNode];
        activeProbs = [activeProbs(keep), treeProb(nextNode)];
        nextNode = nextNode + 1;
    end

    rootNode  = activeNodes(1);
    leafCodes = getLeafCodes(rootNode, treeLeft, treeRight, n);

    dict = cell(n, 2);
    for i = 1:n
        dict{i,1} = symbols(i);
        dict{i,2} = leafCodes{i};
        if isempty(dict{i,2}); dict{i,2} = '0'; end
    end
end

function leafCodes = getLeafCodes(rootNode, treeLeft, treeRight, numLeaves)
    leafCodes = cell(1, numLeaves);
    stack = {rootNode, ''};
    while ~isempty(stack)
        currentNode = stack{1,1};
        currentCode = stack{1,2};
        stack(1,:)  = [];
        if currentNode <= numLeaves
            leafCodes{currentNode} = currentCode;
        else
            stack = [stack; {treeLeft(currentNode),  [currentCode '0']}];
            stack = [stack; {treeRight(currentNode), [currentCode '1']}];
        end
    end
end

function encoded = encodeWithDict(symbols, uniqueSymbols, dict)
% Encode symbol array to binary string using Huffman dict
    parts = cell(1, numel(symbols));
    for i = 1:numel(symbols)
        idx = find(uniqueSymbols == symbols(i), 1);
        if ~isempty(idx)
            parts{i} = dict{idx,2};
        end
    end
    encoded = strjoin(parts, '');
end

function symbols = manualHuffmanDecode(encoded, dict)
% Fast Huffman decode using containers.Map lookup
    lookup = containers.Map('KeyType','char','ValueType','double');
    for d = 1:size(dict,1)
        if ~isempty(dict{d,2})
            lookup(dict{d,2}) = dict{d,1};
        end
    end

    symbols = [];
    i       = 1;
    maxLen  = max(cellfun(@numel, dict(:,2)));

    while i <= numel(encoded)
        matched = false;
        for len = 1:min(maxLen, numel(encoded)-i+1)
            codeword = encoded(i:i+len-1);
            if isKey(lookup, codeword)
                symbols(end+1) = lookup(codeword); %#ok<AGROW>
                i = i + len;
                matched = true;
                break;
            end
        end
        if ~matched; break; end
    end
end