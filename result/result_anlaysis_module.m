clear all

MinMax = load('../MinMax.csv');

ext_jts = load('TestingExtTorque.txt');
data_size = size(load('backward_real_0.csv'),1);
tau_d = load('TestingTauD.txt');

pred = zeros(data_size,7);
real = zeros(data_size,7);


pred(:,1) = MinMax(1,1)*load('backward_prediction_0.csv');
real(:,1) = MinMax(1,1).*load('backward_real_0.csv');

pred(:,2) = MinMax(1,2).*load('backward_prediction_1.csv');
real(:,2) = MinMax(1,2).*load('backward_real_1.csv');

pred(:,3) = MinMax(1,3).*load('backward_prediction_2.csv');
real(:,3) = MinMax(1,3).*load('backward_real_2.csv');

pred(:,4) = MinMax(1,4).*load('backward_prediction_3.csv');
real(:,4) = MinMax(1,4).*load('backward_real_3.csv');

pred(:,5) = MinMax(1,5).*load('backward_prediction_4.csv');
real(:,5) = MinMax(1,5).*load('backward_real_4.csv');

pred(:,6) = MinMax(1,6).*load('backward_prediction_5.csv');
real(:,6) = MinMax(1,6).*load('backward_real_5.csv');

pred(:,7) = MinMax(1,7).*load('backward_prediction_6.csv');
real(:,7) = MinMax(1,7).*load('backward_real_6.csv');


figure();
for i=1:7
    subplot(4,2,i);
    plot(pred(:,i));
    hold on
    plot(real(:,i));
end

resi = pred-real;
err = abs(resi);

figure();
for i=1:7
    subplot(4,2,i);
    plot(resi(:,i));
    hold on
    plot(ext_jts(:,i));
end