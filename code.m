# Hello world
%% Video Compression Project Steps 4 to 7

clear; clc; close all;

%% ============================================================
%% SETUP: Create a synthetic video (if no video file available)
%% ============================================================

fprintf('==> Creating synthetic video frames...\n');

numFrames  = 30;   % total frames
frameH     = 64;   % height (keep small for speed)
frameW     = 64;   % width
blockSize  = 8;    % block size for DCT and motion estimation
searchRange = 4;   % motion estimation search range (pixels)
qStep      = 10;   % quantization step size

% Generate frames: moving white rectangle on black background
frames = zeros(frameH, frameW, numFrames, 'uint8');
for f = 1:numFrames
    img = zeros(frameH, frameW, 'uint8');
    offset = mod(f*2, frameW-20);          % horizontal motion
    img(20:40, offset+1:offset+20) = 200;  % moving block
    img = img + uint8(5*randn(frameH,frameW)); % add slight noise
    frames(:,:,f) = img;
end

fprintf('   Video created: %d frames of %dx%d pixels\n', numFrames, frameH, frameW);

%% ---- Show a sample of the original video ----
figure('Name','ORIGINAL VIDEO - Sample Frames','NumberTitle','off');
sampleIdx = [1, 10, 20, 30];
for k = 1:4
    subplot(1,4,k);
    imshow(frames(:,:,sampleIdx(k)));
    title(sprintf('Frame %d', sampleIdx(k)));
end
sgtitle('Original Video - Sample Frames | عينة من الفيديو الأصلي');


%% ============================================================
%% STEP 4: INTER-FRAME COMPRESSION (P-FRAME)
%% تضغيط الإطارات التنبؤية
%% ============================================================

fprintf('\n==> STEP 4: Inter-frame Compression (P-frames)...\n');

% I-frame every 10 frames, rest are P-frames
isIframe = false(1, numFrames);
isIframe(1:10:end) = true;

% Storage
motionVectors  = cell(1, numFrames);   % {frameIdx} = [numBlocks x 2]
residuals      = cell(1, numFrames);   % {frameIdx} = residual image
frameTypes     = cell(1, numFrames);

%% --- Function handles (defined at bottom as local functions) ---

for f = 1:numFrames
    currentFrame = double(frames(:,:,f));

    if isIframe(f)
        frameTypes{f} = 'I';
        motionVectors{f} = [];
        residuals{f}     = currentFrame;   % I-frame stores full image
        fprintf('   Frame %2d -> I-frame\n', f);
    else
        frameTypes{f} = 'P';
        refFrame = double(frames(:,:,f-1));   % previous frame as reference

        % --- Block Matching Motion Estimation ---
        numBlocksH = frameH / blockSize;
        numBlocksW = frameW / blockSize;
        mvs = zeros(numBlocksH * numBlocksW, 2);  % [dy, dx]
        residualImg = zeros(frameH, frameW);

        blockNum = 0;
        for row = 1:numBlocksH
            for col = 1:numBlocksW
                blockNum = blockNum + 1;

                % Current block coordinates
                rStart = (row-1)*blockSize + 1;
                cStart = (col-1)*blockSize + 1;
                rEnd   = rStart + blockSize - 1;
                cEnd   = cStart + blockSize - 1;

                currentBlock = currentFrame(rStart:rEnd, cStart:cEnd);

                % Search in reference frame
                bestMAD = inf;
                bestDy  = 0;
                bestDx  = 0;

                for dy = -searchRange:searchRange
                    for dx = -searchRange:searchRange
                        rRef = rStart + dy;
                        cRef = cStart + dx;
                        % Boundary check
                        if rRef < 1 || cRef < 1 || ...
                           rRef+blockSize-1 > frameH || cRef+blockSize-1 > frameW
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

                % Compute residual for this block
                rRef = rStart + bestDy;
                cRef = cStart + bestDx;
                predictedBlock = refFrame(rRef:rRef+blockSize-1, cRef:cRef+blockSize-1);
                residualImg(rStart:rEnd, cStart:cEnd) = currentBlock - predictedBlock;
            end
        end

        motionVectors{f} = mvs;
        residuals{f}     = residualImg;
        fprintf('   Frame %2d -> P-frame | Mean MV magnitude: %.2f\n', f, mean(sqrt(sum(mvs.^2,2))));
    end
end

%% ---- Visualize Step 4 Output ----
figure('Name','STEP 4 OUTPUT - Motion Vectors & Residuals','NumberTitle','off');

% Show frame 2 (P-frame) original
subplot(1,3,1);
imshow(frames(:,:,2));
title('Frame 2 - Original | الإطار الأصلي');

% Show residual of frame 2
subplot(1,3,2);
residualDisplay = residuals{2};
imagesc(residualDisplay); colormap gray; colorbar; axis image;
title('Frame 2 - Residual | الباقي');

% Show motion vectors as quiver plot
subplot(1,3,3);
mvs2 = motionVectors{2};
numBlocksH = frameH/blockSize;
numBlocksW = frameW/blockSize;
[bCols, bRows] = meshgrid(1:numBlocksW, 1:numBlocksH);
mvDy = reshape(mvs2(:,1), numBlocksH, numBlocksW);
mvDx = reshape(mvs2(:,2), numBlocksH, numBlocksW);
quiver(bCols, bRows, mvDx, mvDy, 'r');
axis([0 numBlocksW+1 0 numBlocksH+1]);
set(gca,'YDir','reverse');
title('Motion Vectors (Frame 2) | متجهات الحركة');
xlabel('Block Col'); ylabel('Block Row');
sgtitle('Step 4: Inter-frame Compression | تضغيط الإطارات التنبؤية');

%% ============================================================
%% STEP 5: ENTROPY CODING (Manual Huffman - No Toolbox Needed)
%% ============================================================

fprintf('\n==> STEP 5: Entropy Coding (Huffman)...\n');

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

    % Build frequency table
    uniqueSymbols = unique(double(symbols));
    counts = zeros(1, numel(uniqueSymbols));
    for i = 1:numel(uniqueSymbols)
        counts(i) = sum(double(symbols) == uniqueSymbols(i));
    end
    probs = counts / sum(counts);

    % Build Huffman tree manually
    dict = buildHuffmanDict(uniqueSymbols, probs);

    % Encode
    encoded = '';
    symDouble = double(symbols);
    for i = 1:numel(symDouble)
        idx = find(uniqueSymbols == symDouble(i), 1);
        encoded = [encoded, dict{idx,2}]; %#ok<AGROW>
    end

    encodedStream{f}    = encoded;
    huffmanDicts{f}     = dict;
    originalSizeBits(f) = numel(symbols) * 8;
    encodedSizeBits(f)  = numel(encoded);

    fprintf('   Frame %2d [%s] | Original: %5d bits | Encoded: %5d bits | Ratio: %.2f\n', ...
        f, frameTypes{f}, originalSizeBits(f), encodedSizeBits(f), ...
        originalSizeBits(f)/max(encodedSizeBits(f),1));
end

%% ---- Visualize Step 5 Output ----
figure('Name','STEP 5 OUTPUT - Entropy Coding','NumberTitle','off');

subplot(1,2,1);
bar(originalSizeBits, 'FaceColor', [0.2 0.5 0.8]);
hold on;
bar(encodedSizeBits, 'FaceColor', [0.9 0.3 0.3]);
legend('Original Bits | البيانات الأصلية', 'Huffman Encoded | بعد هافمان', 'Location','northeast');
xlabel('Frame Number | رقم الإطار');
ylabel('Bits');
title('Bit Usage per Frame | حجم البيانات لكل إطار');
grid on;

subplot(1,2,2);
compressionRatioPerFrame = originalSizeBits ./ max(encodedSizeBits, 1);
plot(compressionRatioPerFrame, 'g-o', 'LineWidth', 1.5);
xlabel('Frame Number | رقم الإطار');
ylabel('Compression Ratio | نسبة الضغط');
title('Huffman Compression Ratio per Frame | نسبة ضغط هافمان');
grid on;

sgtitle('Step 5: Entropy Coding Results | نتائج الترميز بالإنتروبيا');

%% ============================================================
%% STEP 6: BITSTREAM FORMATION
%% تكوين تدفق البيانات
%% ============================================================

fprintf('\n==> STEP 6: Bitstream Formation...\n');

% We simulate a bitstream as a struct array (in real systems it's binary)
bitstream.header.numFrames  = numFrames;
bitstream.header.frameH     = frameH;
bitstream.header.frameW     = frameW;
bitstream.header.blockSize  = blockSize;
bitstream.header.qStep      = qStep;
bitstream.frames = struct();

totalBitstreamBits = 0;

for f = 1:numFrames
    % Frame header: frameIndex (32b) + type (8b) + dataSize (32b) = 72 bits overhead
    headerBits = 72;
    dataBits   = encodedSizeBits(f);

    bitstream.frames(f).index       = f;
    bitstream.frames(f).type        = frameTypes{f};
    bitstream.frames(f).encodedData = encodedStream{f};
    bitstream.frames(f).huffDict    = huffmanDicts{f};
    bitstream.frames(f).totalBits   = headerBits + dataBits;

    totalBitstreamBits = totalBitstreamBits + headerBits + dataBits;

    fprintf('   Frame %2d [%s] packaged | Frame bits: %d\n', ...
        f, frameTypes{f}, bitstream.frames(f).totalBits);
end

fprintf('   TOTAL Bitstream size: %d bits = %.2f KB\n', ...
    totalBitstreamBits, totalBitstreamBits/8/1024);

rawVideoBytes = frameH * frameW * numFrames;
fprintf('   Raw video size:       %d bytes = %.2f KB\n', ...
    rawVideoBytes, rawVideoBytes/1024);
fprintf('   Overall Compression Ratio: %.2fx\n', ...
    (rawVideoBytes*8) / totalBitstreamBits);

%% ---- Visualize Step 6 Output ----
figure('Name','STEP 6 OUTPUT - Bitstream Formation','NumberTitle','off');

frameBits = arrayfun(@(f) bitstream.frames(f).totalBits, 1:numFrames);
barColors = zeros(numFrames, 3);
for f = 1:numFrames
    if strcmp(bitstream.frames(f).type,'I')
        barColors(f,:) = [0.85 0.2 0.2];   % red for I-frames
    else
        barColors(f,:) = [0.2 0.6 0.85];   % blue for P-frames
    end
end

subplot(1,2,1);
b = bar(frameBits, 'FaceColor','flat');
b.CData = barColors;
xlabel('Frame Index | رقم الإطار');
ylabel('Bits in Bitstream');
title('Bits per Frame in Bitstream | حجم كل إطار في التدفق');
legend([patch(NaN,NaN,[0.85 0.2 0.2]), patch(NaN,NaN,[0.2 0.6 0.85])], ...
    'I-frame','P-frame','Location','northeast');
grid on;

subplot(1,2,2);
labels = {'Raw Video | الفيديو الخام', 'Compressed Bitstream | المضغوط'};
values = [rawVideoBytes*8, totalBitstreamBits];
bar(values, 'FaceColor', [0.4 0.7 0.4]);
set(gca,'XTickLabel', labels);
ylabel('Total Bits');
title(sprintf('Compression Ratio: %.2fx | نسبة الضغط', (rawVideoBytes*8)/totalBitstreamBits));
grid on;

sgtitle('Step 6: Bitstream Formation | تكوين تدفق البيانات');

%% ============================================================
%% STEP 7: TESTING & EVALUATION (Decode + PSNR + Compression Ratio)
%% الاختبار والتقييم
%% ============================================================

fprintf('\n==> STEP 7: Decoding and Evaluation...\n');

decodedFrames = zeros(frameH, frameW, numFrames, 'double');
psnrValues    = zeros(1, numFrames);

for f = 1:numFrames
    frameData = bitstream.frames(f);

    % --- Huffman Decode ---
    decodedSymbols = manualHuffmanDecode(frameData.encodedData, frameData.huffDict);

    if strcmp(frameData.type, 'I')
        % Reconstruct I-frame
        numPixels  = frameH * frameW;
        decodedQuant = decodedSymbols(1:numPixels);
        reconstructed = reshape(double(decodedQuant) * qStep, frameH, frameW);
        reconstructed = max(0, min(255, reconstructed));
        decodedFrames(:,:,f) = reconstructed;

    else
        % Reconstruct P-frame
        numBlocks = (frameH/blockSize) * (frameW/blockSize);
        numMVsymbols = numBlocks * 2;
        numPixels    = frameH * frameW;

        mvDecoded  = double(decodedSymbols(1:numMVsymbols));
        resDecoded = double(decodedSymbols(numMVsymbols+1 : numMVsymbols+numPixels));

        mvMatrix   = reshape(mvDecoded, numBlocks, 2);
        residualRec= reshape(resDecoded * qStep, frameH, frameW);

        % Reconstruct using reference (previous decoded frame)
        refDecoded = decodedFrames(:,:,f-1);
        predicted  = zeros(frameH, frameW);

        numBlocksH = frameH / blockSize;
        numBlocksW = frameW / blockSize;
        blockNum = 0;
        for row = 1:numBlocksH
            for col = 1:numBlocksW
                blockNum = blockNum + 1;
                rStart = (row-1)*blockSize + 1;
                cStart = (col-1)*blockSize + 1;
                rEnd   = rStart + blockSize - 1;
                cEnd   = cStart + blockSize - 1;

                dy = mvMatrix(blockNum,1);
                dx = mvMatrix(blockNum,2);

                rRef = max(1, min(frameH-blockSize+1, rStart+dy));
                cRef = max(1, min(frameW-blockSize+1, cStart+dx));
                predicted(rStart:rEnd, cStart:cEnd) = ...
                    refDecoded(rRef:rRef+blockSize-1, cRef:cRef+blockSize-1);
            end
        end

        reconstructed = predicted + residualRec;
        reconstructed = max(0, min(255, reconstructed));
        decodedFrames(:,:,f) = reconstructed;
    end

    % --- Compute PSNR ---
    orig  = double(frames(:,:,f));
    recon = decodedFrames(:,:,f);
    mse   = mean((orig(:) - recon(:)).^2);
    if mse == 0
        psnrValues(f) = Inf;
    else
        psnrValues(f) = 10 * log10(255^2 / mse);
    end

    fprintf('   Frame %2d [%s] | PSNR: %.2f dB\n', f, frameData.type, psnrValues(f));
end

%% ---- Final Evaluation Summary ----
overallPSNR = mean(psnrValues(isfinite(psnrValues)));
overallCR   = (rawVideoBytes*8) / totalBitstreamBits;

fprintf('\n==========================================\n');
fprintf('  FINAL EVALUATION SUMMARY\n');
fprintf('==========================================\n');
fprintf('  Average PSNR           : %.2f dB\n', overallPSNR);
fprintf('  Compression Ratio      : %.2fx\n', overallCR);
fprintf('  Raw Video Size         : %.2f KB\n', rawVideoBytes/1024);
fprintf('  Compressed Size        : %.2f KB\n', totalBitstreamBits/8/1024);
fprintf('==========================================\n');

%% ---- Visualize Step 7: Side-by-Side Comparison ----
figure('Name','STEP 7 OUTPUT - Original vs Decoded','NumberTitle','off');
showFrames = [1, 2, 10, 11, 20, 21];
for k = 1:min(6, numel(showFrames))
    f = showFrames(k);
    subplot(2,6,k);
    imshow(uint8(frames(:,:,f)));
    title(sprintf('Orig F%d',f), 'FontSize',7);

    subplot(2,6,k+6);
    imshow(uint8(decodedFrames(:,:,f)));
    title(sprintf('Decoded F%d\n%.1fdB',f,psnrValues(f)), 'FontSize',7);
end
sgtitle('Step 7: Original vs Decoded | الأصلي مقابل المُستعاد');

%% ---- PSNR Plot ----
figure('Name','STEP 7 OUTPUT - PSNR per Frame','NumberTitle','off');

subplot(1,2,1);
finiteIdx  = isfinite(psnrValues);
plot(find(finiteIdx), psnrValues(finiteIdx), 'b-o', 'LineWidth',1.5);
hold on;
yline(30, 'r--', 'Acceptable (30 dB)', 'LabelHorizontalAlignment','left');
yline(40, 'g--', 'Excellent (40 dB)',  'LabelHorizontalAlignment','left');
iFrameIdx = find(isIframe & finiteIdx);
plot(iFrameIdx, psnrValues(iFrameIdx), 'r^', 'MarkerSize',8, 'DisplayName','I-frames');
xlabel('Frame Number | رقم الإطار');
ylabel('PSNR (dB)');
title(sprintf('PSNR per Frame | جودة كل إطار\nMean: %.2f dB', overallPSNR));
legend('PSNR','30dB threshold','40dB threshold','I-frames','Location','southeast');
grid on;

subplot(1,2,2);
pieLabels = {sprintf('Compressed\n%.2f KB', totalBitstreamBits/8/1024), ...
             sprintf('Saved Space\n%.2f KB', (rawVideoBytes - totalBitstreamBits/8)/1024)};
pie([totalBitstreamBits/8, rawVideoBytes - totalBitstreamBits/8], pieLabels);
title(sprintf('Compression Ratio: %.2fx | نسبة الضغط', overallCR));
colormap([0.9 0.3 0.3; 0.3 0.7 0.3]);

sgtitle('Step 7: Evaluation - PSNR & Compression | التقييم النهائي');

fprintf('\n==> ALL STEPS COMPLETE! Check the figures.\n');


%% ============================================================
%% HELPER FUNCTIONS - Huffman (No Toolbox)
%% ============================================================

function dict = buildHuffmanDict(symbols, probs)
% Build Huffman dictionary without any toolbox
n = numel(probs);

if n == 1
    dict = {symbols(1), '0'};
    return;
end

% Each node: {probability, symbol_indices}
nodes = cell(n, 1);
nodeProbs = probs;
for i = 1:n
    nodes{i} = i;  % leaf node pointing to symbol index
end

% Merge nodes until one tree remains
codes = cell(n, 1);
for i = 1:n
    codes{i} = '';
end

allProbs  = nodeProbs;
allNodes  = nodes;
numNodes  = n;
treeLeft  = zeros(1, 2*n);
treeRight = zeros(1, 2*n);
treeProb  = zeros(1, 2*n);
treeProb(1:n) = probs;
nextNode  = n + 1;

activeNodes = 1:n;
activeProbs = probs;

while numel(activeNodes) > 1
    % Find two smallest probabilities
    [sorted, sortIdx] = sort(activeProbs);
    i1 = sortIdx(1);
    i2 = sortIdx(2);

    % Merge into new node
    treeLeft(nextNode)  = activeNodes(i1);
    treeRight(nextNode) = activeNodes(i2);
    treeProb(nextNode)  = sorted(1) + sorted(2);

    % Remove i1 and i2, add new node
    keep = setdiff(1:numel(activeNodes), [i1 i2]);
    activeNodes = [activeNodes(keep), nextNode];
    activeProbs = [activeProbs(keep), treeProb(nextNode)];
    nextNode = nextNode + 1;
end

rootNode = activeNodes(1);

% Traverse tree to assign codes
leafCodes = cell(1, n);
traverseTree(rootNode, '', treeLeft, treeRight, n, leafCodes);

% We need to get leafCodes out - use persistent workaround
leafCodes = getLeafCodes(rootNode, treeLeft, treeRight, n);

dict = cell(n, 2);
for i = 1:n
    dict{i,1} = symbols(i);
    dict{i,2} = leafCodes{i};
end
end

function leafCodes = getLeafCodes(node, treeLeft, treeRight, numLeaves)
leafCodes = cell(1, numLeaves);
stack = {node, ''};
while ~isempty(stack)
    currentNode = stack{1,1};
    currentCode = stack{1,2};
    stack(1,:) = [];

    if currentNode <= numLeaves
        % Leaf node
        leafCodes{currentNode} = currentCode;
    else
        % Internal node - push children
        stack = [stack; {treeLeft(currentNode),  [currentCode '0']}];
        stack = [stack; {treeRight(currentNode), [currentCode '1']}];
    end
end
end

function traverseTree(~, ~, ~, ~, ~, ~)
% placeholder - not used directly
end

function symbols = manualHuffmanDecode(encoded, dict)
    % Build a fast lookup: codeword_string -> symbol value
    lookup = containers.Map('KeyType','char','ValueType','double');
    for d = 1:size(dict,1)
        lookup(dict{d,2}) = dict{d,1};
    end

    symbols = [];
    i = 1;
    maxLen = max(cellfun(@numel, dict(:,2)));  % max codeword length

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
        if ~matched
            break; % safety exit
        end
    end
end

