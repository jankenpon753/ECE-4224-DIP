clc;
clear;
close all;

% Step 1: Define image A
A = [0 0 0 0 0 0 0;
     0 1 1 1 1 1 0;
     0 1 0 0 0 1 0;
     0 1 0 0 0 1 0;
     0 1 0 0 0 1 0;
     0 1 1 1 1 1 0;
     0 0 0 0 0 0 0];

A = logical(A);

% Step 2: Complement
Ac = ~A;

% Step 3: Structuring Element
B = [0 1 0;
     1 1 1;
     0 1 0];

% Step 4: Seed point
X_prev = zeros(size(A));
X_prev(4,4) = 1;

% Step 5: Iteration
while true
    X_dilated = imdilate(X_prev, B);
    X_new = X_dilated & Ac;

    if isequal(X_new, X_prev)
        break;
    end

    X_prev = X_new;
end

Xk = X_new;

% Step 6: Final result
Filled = A | Xk;

% Step 7: Display
figure;

subplot(2,3,1);
showMatrix(A, 'Original Boundary A');

subplot(2,3,2);
showMatrix(Ac, 'Complement A^c');

subplot(2,3,3);
showMatrix(X_prev, 'Initial Seed X0');

subplot(2,3,4);
showMatrix(Xk, 'Filled Region Xk');

subplot(2,3,5);
showMatrix(Filled, 'Final Filled Image');

subplot(2,3,6);
showMatrix(B, 'Structuring Element');


%% -------- LOCAL FUNCTION (MUST BE AT END) --------
function showMatrix(mat, titleText)
    imagesc(mat);
    colormap(gray);
    axis equal;
    axis tight;
    title(titleText);

    [r,c] = size(mat);
    xticks(1:c);
    yticks(1:r);
    grid on;

    for i = 1:r
        for j = 1:c
            text(j,i,num2str(mat(i,j)),...
                'HorizontalAlignment','center',...
                'Color','red','FontWeight','bold');
        end
    end

    set(gca,'XTickLabel',[],'YTickLabel',[]);
end