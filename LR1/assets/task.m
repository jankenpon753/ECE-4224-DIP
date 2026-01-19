clc;
clear;
close all;

%% 1. Read Image and Convert
% Read the image
rgb_img = imread('Lenna.png');

% Convert to Grayscale
gray_img = rgb2gray(rgb_img);

% Display Step 1
figure; imshow(rgb_img); title('Step 1: Original RGB Image');
figure; imshow(gray_img); title('Step 1: Grayscale Image');

%% 2. Identify Objects (Red Channel > 150)
% Extract Red Channel
R = rgb_img(:,:,1);

% Create logical mask where Red intensity > 150
mask = R > 150;

% Display Step 2
figure; imshow(mask); title('Step 2: Binary Mask (Red > 150)');

%% 3. Create Masked Image (Isolate Object)
% Initialize masked image with zeros (black)
masked_img = zeros(size(rgb_img), 'uint8');

% Apply mask to all three channels
masked_img(:,:,1) = rgb_img(:,:,1) .* uint8(mask);
masked_img(:,:,2) = rgb_img(:,:,2) .* uint8(mask);
masked_img(:,:,3) = rgb_img(:,:,3) .* uint8(mask);

% Display Step 3
figure; imshow(masked_img); title('Step 3: Isolated Masked Object');

%% 4. Apply Transformations on Masked Object
% i. Flip Horizontally
masked_flip = fliplr(masked_img);

% ii. Rotate 90 degrees clockwise
% Note: rot90 rotates counter-clockwise by default, so we use -1 for clockwise
masked_rot_raw = rot90(masked_flip, -1);

% Resize back to original image dimensions to allow merging
% (Rotation changes dimensions, so we force it back to fit the frame)
masked_rot = imresize(masked_rot_raw, [size(rgb_img,1), size(rgb_img,2)]);

% iii. Increase intensity of dominant channel (Red) by 50%
% We convert to double for calculation, multiply by 1.5, then clip at 255
boosted_red = double(masked_rot(:,:,1)) * 1.5;
masked_rot(:,:,1) = uint8(min(boosted_red, 255));

% Display Step 4
figure; imshow(masked_rot); title('Step 4: Transformed Object (Flipped, Rotated, Boosted)');

%% 5. Merge Transformed Object Back to Original
% Logic: Keep the background where the mask is NOT present (~mask),
% then add the new transformed object (masked_rot).
final_img = rgb_img;

% Clear the area where the original object was (create a "hole")
background_only = rgb_img .* uint8(repmat(~mask, [1, 1, 3]));

% Add the transformed object to the background
final_img = background_only + masked_rot;

% Display Step 5
figure; imshow(final_img); title('Step 5: Final Reconstructed Image');

%% 6. Compute Area
% Sum of all white pixels in the binary mask
object_area = sum(mask(:));

% Display Area in Command Window
fprintf('--------------------------------------------------\n');
fprintf('Object Detection Analysis:\n');
fprintf('Total Area of Detected Objects: %d pixels\n', object_area);
fprintf('--------------------------------------------------\n');

%% 7. Summary Display (All Results)
% Create a single figure with subplots for the report
figure('Name', 'Conceptual Task Results', 'NumberTitle', 'off');

subplot(2, 3, 1);
imshow(rgb_img);
title('Original Image');

subplot(2, 3, 2);
imshow(mask);
title('Mask (Red > 150)');

subplot(2, 3, 3);
imshow(masked_img);
title('Isolated Object');

subplot(2, 3, 4);
imshow(masked_rot);
title('Transformed Object');

subplot(2, 3, 5);
imshow(final_img);
title('Final Reconstructed');

subplot(2, 3, 6);
% Display text for the area
axis off;
text(0.1, 0.5, sprintf('Detected Area:\n%d pixels', object_area), ...
     'FontSize', 12, 'FontWeight', 'bold');
title('Area Calculation');