clear all

train = load('training_result.txt');
test = load('testing_result.txt');

figure();

plot(train)
hold on
plot(test)
