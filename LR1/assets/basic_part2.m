%% 1. Merge Image
% Concatenate Grayscale (c) and Binary (d)
% We multiply logical 'd' by 255 and convert to uint8 to match 'c'
m = horzcat(c, uint8(d) * 255); 
figure; 
imshow(m); 
title('Merged Image (Grayscale + Binary)');

% 2. Channel Manipulation
% Create a copy to avoid modifying original 'a'
a_mod = a; 
a_mod(:, :, 1) = 0; % Set Red channel to 0
a_mod(:, :, 2) = 0; % Set Green channel to 0
figure; 
imshow(a_mod); 
title('Blue Channel Only');

% 3. Flip and Rotate
o = flip(a, 2);      % Horizontal flip
p = imrotate(a, 90); % 90 degree clockwise rotation
figure; imshow(o); title('Flipped Horizontally');
figure; imshow(p); title('Rotated 90 Degrees');

% 4. Change Color Intensity
red_factor = 0.2;
green_factor = 1.8;
blue_factor = 0.5;

% Scale individual channels (using 'a' as source)
red_channel   = uint8(a(:, :, 1) * red_factor);
green_channel = uint8(a(:, :, 2) * green_factor);
blue_channel  = uint8(a(:, :, 3) * blue_factor);

% Recombine channels
modified_img = cat(3, red_channel, green_channel, blue_channel);

figure; 
imshow(modified_img); 
title('Color Intensity Modified');

%% --- SUMMARY DISPLAY (All 10 Results) ---
figure('Name', 'Basic Operations Complete Summary', 'NumberTitle', 'off', 'Position', [50, 50, 1400, 600]);

% Row 1: Basic Conversions
subplot(2, 5, 1); imshow(a); title('1. Original');
subplot(2, 5, 2); imshow(b); title('2. Resized (50x50)');
subplot(2, 5, 3); imshow(c); title('3. Grayscale');
subplot(2, 5, 4); imshow(d); title('4. Binary');
subplot(2, 5, 5); imshow(m); title('5. Merged (Gray+Bin)');

% Row 2: Manipulations
subplot(2, 5, 6); imshow(a_mod); title('6. Blue Channel Only');
subplot(2, 5, 7); imshow(n); title('7. Flip Vertical');
subplot(2, 5, 8); imshow(o); title('8. Flip Horizontal');
subplot(2, 5, 9); imshow(p); title('9. Rotated 90\circ');
subplot(2, 5, 10); imshow(modified_img); title('10. Color Modified');