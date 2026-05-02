clc;
clear;

%% Read Video
video = VideoReader('WhatsApp Video 2026-05-02 at 9.12.50 PM.mp4');

frames = {};
yuv_frames = {};

i = 1;

while hasFrame(video)
    frame = readFrame(video);
    frames{i} = frame;

    yuv = rgb2ycbcr(frame);
    yuv_frames{i} = yuv;

    i = i + 1;
end

numFrames = length(frames);
disp(['Total Frames: ', num2str(numFrames)]);
for i = 1:30
    imshow(yuv_frames{i});
    title(['Frame ', num2str(i)]);
     pause(0.2);
end
%% Frame Types
frame_types = strings(1, numFrames);

for i = 1:numFrames
    if mod(i,10) == 1
        frame_types(i) = "I";
    else
        frame_types(i) = "P";
    end
    
end
disp(frame_types);

%% Compression (I-frames only)

Q = ones(8,8) * 10;
compressed_data = {};

for i = 1:numFrames

    if frame_types(i) == "I"

        frame = yuv_frames{i};
        Y = double(frame(:,:,1));

        [h, w] = size(Y);

        compressed_frame = {};
        idx = 1;

        for x = 1:8:h-7
            for y = 1:8:w-7

                block = Y(x:x+7, y:y+7);

                dct_block = dct2(block);
                q_block = round(dct_block ./ Q);

                flat = q_block(:)';
                encoded = rle_encode(flat);

                %%results
                compressed_frame{idx} = encoded;
                idx = idx + 1;

            end
        end

        compressed_data{i} = compressed_frame;

    end

end
disp('First I-frame compressed data:');
disp(compressed_data{1}{1});
original_block = Y(1:8,1:8);

figure;
subplot(1,2,1);
imshow(original_block, []);
title('Original Block');

subplot(1,2,2);
imshow(q_block, []);
title('Compressed Block');

%% RLE
function encoded = rle_encode(arr)

encoded = [];
count = 1;

for i = 2:length(arr)
    if arr(i) == arr(i-1)
        count = count + 1;
    else
        encoded = [encoded; arr(i-1), count];
        count = 1;
    end
end

encoded = [encoded; arr(end), count];

end

disp(encoded);