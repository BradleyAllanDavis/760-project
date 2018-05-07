

%%%%% CNN Results
cnn_train_size = [1e3,5e3,1e4,2e4,3e4,4e4,5e4,6e4];
cnn_test_accuracy = [0.767,0.928,0.9639,0.9774,0.9846,0.985,0.987,0.9894];


%%%%% CapsNet EM Results
em_train_size = [1e3,5e3,1e4,3e4,6e4];
em_test_accuracy = [0.315,0.966,0.988,0.9930,0.9950];


close all;

clf;
hold on;
plot(cnn_train_size,cnn_test_accuracy,'-o','linewidth',2);
plot(em_train_size,em_test_accuracy,'-o','linewidth',2);

grid on;
title('Learning Curve: CapsNet-EM and CNN','FontSize',15);
xlabel('Num Training Samples','FontSize',15)
ylabel('Test Accuracy','FontSize',15)
h = legend('CNN','CapsNet-EM','location','southeast');
set(h,'FontSize',14); 


