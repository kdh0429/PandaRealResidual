clear all

MinMax = load('../data/MinMax.csv');

pred = MinMax(1,:).*load('backward_prediction.csv');
real = MinMax(1,:).*load('backward_real.csv');
tau_d = load('TestingTauD.txt');

ext_jts = load('TestingExtTorque.txt');

t=0.01:0.01:0.01*size(pred,1);

figure();
for i=1:7
    subplot(4,2,i);
    plot(t,real(:,i));
    hold on
    plot(t,pred(:,i));
end

resi = real-pred;
err = abs(resi);

csvwrite('TestingResi.csv',resi);

resi_lpw = lowpass(resi, 20, 100);

figure();
for i=1:7
    subplot(4,2,i);
    plot(t,ext_jts(1:size(resi,1),i));
    hold on
    %plot(t,resi_lpw(:,i));
    plot(t,resi(:,i));
end

mean(err)

% 1 hour of data
% 0.608304792531861   0.772250758025253   0.649209578330087   0.539564591641582   0.250543640975170   0.210775351045692   0.081644451141844
% 2 hours of data
% 0.531189658157883   0.624927376490097   0.520639200474754   0.416328411783372   0.171901400461239   0.159770573351461   0.068054442637222
% 3 hours of data
% 0.518466399732580   0.574314275090446   0.471509730215529   0.384223670550695   0.175991380679345   0.148805704024193   0.066666844557475

% New dataset (3 hours)
% 0.432876969236735   0.539886487964876   0.440508337934324   0.371930080894444   0.159456151134968   0.157271102960925   0.059604862323421
% JTS
% 0.356946549775113   0.525223208436644   0.418584999173787   0.563352637249312   0.421018840966464   0.394191367906058   0.213948542576373