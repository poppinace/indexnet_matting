
clear; close all; clc

GT_DIR = '/media/hao/DATA/Combined_Dataset';
RE_DIR = '/home/hao/Pytorch_Codes/ICCV19 Matting Results/Baseline/mobilenet_width_mult_1dot4_unet_decoder_std_conv5x5_stride32';
DATA_TEST_LIST = '../test.txt';

fid = fopen(DATA_TEST_LIST);
imlist = textscan(fid, '%s%s%s', 'Delimiter','\t');
fclose(fid);

sad = zeros(length(imlist{1}), 1);
mse = zeros(length(imlist{1}), 1);
grad = zeros(length(imlist{1}), 1);
conn = zeros(length(imlist{1}), 1);
parfor i = 1:length(imlist{1})
  [~, imname, ~] = fileparts(imlist{1}{i});
  
  pd = imread(fullfile(RE_DIR, [imname '.png']));
  gt = imread(fullfile(GT_DIR, imlist{2}{i}));
  tr = imread(fullfile(GT_DIR, imlist{3}{i}));
  
  gt = gt(:, :, 1);
  
  sad(i) = compute_sad_loss(pd, gt, tr);
  mse(i) = compute_mse_loss(pd, gt, tr);
  grad(i) = compute_gradient_loss(pd, gt, tr) / 1e3;
  conn(i) = compute_connectivity_error(pd, gt, tr, 0.1) / 1e3;
  
  fprintf('test: %d\n', i)
end

SAD = mean(sad);
MSE = mean(mse);
GRAD = mean(grad);
CONN = mean(conn);

fprintf('SAD: %.2f, MSE: %.4f, Grad: %.2f, Conn: %.2f\n', ...
  SAD, MSE, GRAD, CONN)

