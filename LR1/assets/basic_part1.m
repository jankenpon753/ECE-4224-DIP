clc; clear all; close all;

% 1. Read Image
a = imread('Lenna.png');
figure; 
imshow(a); 
title('Original Image');

% 2. Resize
% Resize to 50x50 pixels
b = imresize(a, [50, 50]);
figure; 
imshow(b); 
title('Resized Image');

% 3. Image Info
info = imfinfo('Lenna.png');
disp(info);

% 4. Convert to Grayscale
c = rgb2gray(a);
figure; 
imshow(c); 
title('Grayscale');

% 5. Convert to Binary
% Note: im2bw is older; imbinarize is recommended for newer MATLAB versions
d = im2bw(a); 
figure; 
imshow(d); 
title('Binary Image');