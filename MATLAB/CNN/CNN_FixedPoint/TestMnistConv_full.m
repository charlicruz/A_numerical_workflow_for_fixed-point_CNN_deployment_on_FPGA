clear all
clc

% -------------------------------------------------------------------------
% Add fixed-point generated code folders to MATLAB path
% -------------------------------------------------------------------------

addpath(baseFixptPath);
addpath(genpath(fullfile(baseFixptPath, 'codegen')));

% -------------------------------------------------------------------------
% Load trained weights
% -------------------------------------------------------------------------
load('W1.mat','W1')
load('W5.mat','W5')
load('Wo.mat','Wo')

% -------------------------------------------------------------------------
% Load full official MNIST test dataset
% -------------------------------------------------------------------------
if exist('t10k-images.idx3-ubyte', 'file')
    testImagesFile = 't10k-images.idx3-ubyte';
elseif exist('t10k-images-idx3-ubyte', 'file')
    testImagesFile = 't10k-images-idx3-ubyte';
else
    error(['MNIST test image file not found. ', ...
           'Please place t10k-images.idx3-ubyte or ', ...
           't10k-images-idx3-ubyte in the current MATLAB folder.']);
end

if exist('t10k-labels.idx1-ubyte', 'file')
    testLabelsFile = 't10k-labels.idx1-ubyte';
elseif exist('t10k-labels-idx1-ubyte', 'file')
    testLabelsFile = 't10k-labels-idx1-ubyte';
else
    error(['MNIST test label file not found. ', ...
           'Please place t10k-labels.idx1-ubyte or ', ...
           't10k-labels-idx1-ubyte in the current MATLAB folder.']);
end

TestImages = loadMNISTImages(testImagesFile);
TestImages = reshape(TestImages, 28, 28, []);

TestLabels = loadMNISTLabels(testLabelsFile);
TestLabels(TestLabels == 0) = 10;    % 0 --> 10 for MATLAB indexing

% -------------------------------------------------------------------------
% Test with the complete MNIST test dataset
% -------------------------------------------------------------------------
X = TestImages;
D = TestLabels;

N = length(D);

fprintf('Number of test images loaded: %d\n', N);

if N ~= 10000
    warning('The loaded test dataset does not contain 10000 images. Please check the MNIST files.');
end

cost = zeros(N,1);
predictedLabels = zeros(N,1);
acc = 0;

for k = 1:N
    x = X(:, :, k);                         % Input, 28x28

    y1 = Conv_wrapper_fixpt(x, W1);         % Convolution, 20x20x20
    y2 = ReLU1_wrapper_fixpt(y1);
    y3 = Pool_wrapper_fixpt(y2);            % Pooling, 10x10x20
    y4 = reshape(y3, [], 1);                % Flatten, 2000x1

    v5 = mult1_wrapper_fixpt(W5, y4);       % Fully connected, 100x1
    y5 = ReLU2_wrapper_fixpt(v5);

    v  = mult2_wrapper_fixpt(Wo, y5);       % Output layer, 10x1
    y  = Softmax_wrapper_fixpt(v);          % Softmax, 10x1

    d = zeros(10, 1);
    d(D(k)) = 1;

    cost(k,1) = ((norm(double(d) - double(y)))^2)/2;

    [~, i] = max(y);
    predictedLabels(k) = i;

    if i == D(k)
        acc = acc + 1;
    end

    if mod(k,1000) == 0
        fprintf('Processed %d / %d images\n', k, N);
    end
end

accuracy = acc / N;
accuracyPercent = accuracy * 100;

fprintf('\nFixed-point accuracy on the full MNIST test set is %.4f %%\n', accuracyPercent);
fprintf('Correct classifications: %d / %d\n', acc, N);

% -------------------------------------------------------------------------
% Convert labels back from MATLAB indexing
% Class 10 corresponds to digit 0
% -------------------------------------------------------------------------
trueDigits = D;
predictedDigits = predictedLabels;

trueDigits(trueDigits == 10) = 0;
predictedDigits(predictedDigits == 10) = 0;

% -------------------------------------------------------------------------
% Confusion matrix
% -------------------------------------------------------------------------
figure()
confusionchart(trueDigits, predictedDigits);
title('Fixed-point CNN confusion matrix on full MNIST test set');

% -------------------------------------------------------------------------
% Plot cost
% -------------------------------------------------------------------------
figure()
plot(1:N, cost)
xlabel('Test image index')
ylabel('Cost')
title('Fixed-point CNN cost over the full MNIST test set')
grid on

% -------------------------------------------------------------------------
% Save test results
% -------------------------------------------------------------------------
save('FixedPoint_Full_MNIST_Test_Results.mat', ...
     'accuracy', ...
     'accuracyPercent', ...
     'acc', ...
     'N', ...
     'cost', ...
     'trueDigits', ...
     'predictedDigits');