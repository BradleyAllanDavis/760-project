
%%%%% Load all mnist data
filepath_mnist_dr = 'landmark_capsnetDR/epoch_accuracy_results_mnist.csv';
accuracy_mnist_dr = csvread(filepath_mnist_dr);
filepath_mnist_em = 'EM/data/accuracy_epochs_capsnet_em_mnist.txt';
accuracy_mnist_em = dlmread(filepath_mnist_em);


%%%%% Load all landmark data
filepath_landmark_dr = 'landmark_capsnetDR/epoch_accuracy_results_landmark.csv';
accuracy_landmark_dr = csvread(filepath_landmark_dr);
filepath_landmark_em = 'EM/data/accuracy_epochs_capsnet_em_landmark.txt';
accuracy_landmark_em = dlmread(filepath_landmark_em);



close all;

clf;
hold on;
plot(accuracy_mnist_dr(2:end,1)+2,accuracy_mnist_dr(2:end,2),'linewidth',2);
plot(accuracy_mnist_em(:,1),accuracy_mnist_em(:,2),'linewidth',2);
grid on;
title('Test Accuracy MNIST','FontSize',15);
xlabel('Epoch','FontSize',15)
ylabel('Accuracy','FontSize',15)
h = legend('CapsNet-DR','CapsNet-EM','location','southeast');
set(h,'FontSize',14); 


figure;
hold on;
plot(accuracy_landmark_dr(:,1),accuracy_landmark_dr(:,2),'linewidth',2);
plot(accuracy_landmark_em(:,1),accuracy_landmark_em(:,2),'linewidth',2);
grid on;
title('Test Accuracy Landmark','FontSize',15);
xlabel('Epoch','FontSize',15)
ylabel('Accuracy','FontSize',15)
h = legend('CapsNet-DR','CapsNet-EM','location','southeast');
set(h,'FontSize',14);










