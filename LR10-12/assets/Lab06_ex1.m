clc;
clear;
close all;

% Step 1: Define the binary image A
A = [0 0 0 0 0 0;
     0 1 1 1 1 0;
     0 1 1 1 1 0;
     0 1 1 1 1 0;
     0 1 1 1 1 0;
     0 0 0 0 0 0];

% Step 2: Convert to logical (binary image)
A = logical(A);

% Step 3: Define Structuring Element B (3x3 ones)
B = ones(3,3);

% Step 4: Perform erosion (A ⊖ B)
C = imerode(A, B);

% Step 5: Boundary extraction
boundary = A - C;

% Step 6: Display results
figure;

subplot(1,3,1);
imshow(A);
title('Original Image A');

subplot(1,3,2);
imshow(C);
title('Eroded Image (A ⊖ B)');

subplot(1,3,3);
imshow(boundary);
title('Boundary = A - (A ⊖ B)');