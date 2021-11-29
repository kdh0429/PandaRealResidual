clear all

MinMax = load('../MinMax.csv');

pred = MinMax(1,:).*load('backward_prediction.csv');
real = MinMax(1,:).*load('backward_real.csv');
tau_d = load('TestingTauD.txt');

ext_jts = load('TestingExtTorque.txt');

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