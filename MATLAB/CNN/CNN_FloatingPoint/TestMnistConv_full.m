clear all
clc

% Load training data
Images = loadMNISTImages('train-images.idx3-ubyte');
Images = reshape(Images, 28, 28, []);

Labels = loadMNISTLabels('train-labels.idx1-ubyte');
Labels(Labels == 0) = 10;    % 0 --> 10

rng(1);

% Initialize weights
W1 = 1e-2*randn([9 9 20]);
W5 = (2*rand(100, 2000) - 1) * sqrt(6) / sqrt(100 + 2000);
Wo = (2*rand(10, 100) - 1) * sqrt(6) / sqrt(10 + 100);

% Training subset
X = Images(:, :, 1:8000);
D = Labels(1:8000);

for epoch = 1:23
  fprintf('Epoch %d\n', epoch);
  [W1, W5, Wo] = MnistConv(W1, W5, Wo, X, D);
end

save('MnistConv.mat','W1','W5','Wo');

% Load trained weights
load('MnistConv.mat','W1','W5','Wo');

% Load full MNIST test set
TestImages = loadMNISTImages('t10k-images-idx3-ubyte');
TestImages = reshape(TestImages, 28, 28, []);

TestLabels = loadMNISTLabels('t10k-labels-idx1-ubyte');
TestLabels(TestLabels == 0) = 10;

X = TestImages;
D = TestLabels;

N = length(D);
cost = zeros(N,1);
acc = 0;

for k = 1:N
  x = X(:, :, k);

  y1 = Conv(x, W1);
  y2 = ReLU(y1);
  y3 = Pool(y2);
  y4 = reshape(y3, [], 1);

  v5 = W5*y4;
  y5 = ReLU(v5);

  v  = Wo*y5;
  y  = Softmax(v);

  d = zeros(10, 1);
  d(D(k)) = 1;

  cost(k) = ((norm(d-y))^2)/2;

  [~, i] = max(y);

  if i == D(k)
    acc = acc + 1;
  end
end

acc = acc / N;
fprintf('Accuracy on the full MNIST test set is %.4f %%\n', acc*100);

figure()
plot(1:N, cost)
xlabel('Test image index')
ylabel('Cost')
title('Cost over the full MNIST test set')
grid on