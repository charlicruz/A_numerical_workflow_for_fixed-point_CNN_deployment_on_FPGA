clear all

Images = loadMNISTImages('t10k-images-idx3-ubyte');
Images = reshape(Images, 28, 28, []);
Labels = loadMNISTLabels('./t10k-labels-idx1-ubyte');
Labels(Labels == 0) = 10;    % 0 --> 10

% rng(1);
% 
% % Learning
% %
% W1 = 1e-2*randn([9 9 20]);
% W5 = (2*rand(100, 2000) - 1) * sqrt(6) / sqrt(360 + 2000);
% Wo = (2*rand( 10,  100) - 1) * sqrt(6) / sqrt( 10 +  100);
% 
% X = Images(:, :, 1:8000);
% D = Labels(1:8000);
% 
% for epoch = 1:2
%   epoch
%   [W1, W5, Wo] = MnistConv(W1, W5, Wo, X, D);
% end

save('MnistConv.mat'); 

load('W1.mat','W1')
load('W5.mat','W5')
load('Wo.mat','Wo')


% Test
%
X = Images(:, :, 1:10000);
D = Labels(1:10000);

cost=zeros(10000,1);
acc = 0;
N   = length(D);

for k = 1:N
  x = X(:, :, k);                   % Input,           28x28

  y1 = Conv_wrapper_fixpt(x, W1);           % Convolution,  20x20x20
  y2 = ReLU1_wrapper_fixpt(y1);                    %
  y3 = Pool_wrapper_fixpt(y2);                    % Pool,         10x10x20
  y4 = reshape(y3, [], 1);          %                   2000  
  v5 = mult1_wrapper_fixpt(W5,y4);                       % ReLU,              360
  y5 = ReLU2_wrapper_fixpt(v5);                    %
  v  = mult2_wrapper_fixpt(Wo,y5);  % Softmax,            10
  y  = Softmax_wrapper_fixpt(v);                  %

  d = zeros(10, 1);
  d(sub2ind(size(d), D(k), 1)) = 1;
  cost(k,1)=((norm(double(d)-double(y)))^2)/2;

  [~, i] = max(double(y));
  if i == D(k)
    acc = acc + 1;
  end
end

ejecost=linspace(1,10000,10000);
p=fit(ejecost', cost,'exp1');
figure()
plot(p,ejecost',cost)

acc = acc / N;
fprintf('Accuracy is %f\n', acc);
fprintf('Accuracy percentage is %.4f %%\n', acc*100);