%%importing data
opts = detectImportOptions('training_set_hr.xlsx');
data = readtable('training_set_hr.xlsx',opts);
%%data pre-processing 
%dummy var
data.sales = nominal(data.sales);
dv_sales= dummyvar(data.sales);
data.salary = nominal(data.salary);
dv_salary= dummyvar(data.salary);%% salary
%standraization
zlast_evaluation = zscore(data.last_evaluation);
zsatisfaction_level = zscore(data.satisfaction_level);
z=zscore([ data.time_spend_company data.average_montly_hours data.number_project]);
no_project= z(:,3);%%number_project
years_spend= z(:,1);%%time_spend_company
avg_mothly_hours= z(:,2); %%average_montly_hours
%input x $ output y
x=[ zlast_evaluation zsatisfaction_level  no_project years_spend avg_mothly_hours data.promotion_last_5years data.Work_accident dv_salary(:,1:2) dv_sales(:,1:9)];
y= data.left;
%test_data processing
opts = detectImportOptions('test_set_hr.xls');
test_data = readtable('test_set_hr.xls',opts);
%test data set normalizeation
test_data.sales = nominal(test_data.sales);
dv_sales= dummyvar(test_data.sales);
test_data.salary = nominal(test_data.salary);
dv_salary= dummyvar(test_data.salary);%% salary
zlast_evaluation = zscore(test_data.last_evaluation);
zsatisfaction_level = zscore(test_data.satisfaction_level);
z=zscore([ test_data.time_spend_company test_data.average_montly_hours test_data.number_project]);
no_project= z(:,3);%%number_project
years_spend= z(:,1);%%time_spend_company
avg_mothly_hours= z(:,2); %%average_montly_hours
%new x & y
test_x=[ zlast_evaluation zsatisfaction_level  no_project years_spend avg_mothly_hours test_data.promotion_last_5years test_data.Work_accident dv_salary(:,1:2) dv_sales(:,1:9)];
test_y=test_data.left;
%  transpose data
y =  y';
y =  [~y; y];
x =x';
test_x=test_x';
test_y(test_y~=0) =  1;
test_y =  test_y';
test_y =  [~test_y; test_y];

%autoencoder layer1
hiddenSize1 = 10;
autoenc1 = trainAutoencoder(x,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

% Extract the features in the hidden layer.
feat1 = encode(autoenc1,x);
%autoencoder layer2
hiddenSize2 = 5;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);
% Extract the features in the hidden layer.
feat2 = encode(autoenc2,feat1);

%autoencoder  for test data
hiddenSize1 = 5;
autoenc1 = trainAutoencoder(test_x,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

% Extract the features in the hidden layer.
test_feat1 = encode(autoenc1,test_x);
%SVM
fx=feat2';
y= data.left;
svm_mdl = fitcsvm(fx, y, 'KernelFunction', 'rbf', 'Crossval', 'on')

%%confusion matrix for class 1 (left)
test_y=test_data.left;
test_feat1=test_feat1';
[label, score] = predict(svm_mdl.Trained{3,1},test_feat1 );
CFM = confusionmat(test_y, label)
%accuracy, precsion, recall for class 0
accuracy = sum(diag(CFM))/sum(CFM(:))
precision= CFM(1,1)/sum(CFM(:,1))
recall = CFM(1,1)/sum(CFM(1,:))
f=(2*precision*recall)/(precision+recall)
%accuracy, precsion, recall for class 1
precision1= CFM(2,2)/sum(CFM(:,2))
recall1 = CFM(2,2)/sum(CFM(2,:))
f1=(2*precision1*recall1)/(precision1+recall1)
%%ans2
[xpos1, ypos1, T1, AUC1] = perfcurve(test_y, score(:,1), 0);
figure, plot(xpos1, ypos1)
xlim([-0.05 1.05]), ylim([-0.05 1.05])
xlabel('\bf FP rate'), ylabel('\bf TP rate')
title('\bf ROC for Classification by SVM_class 0')

[xpos, ypos, T, AUC] = perfcurve(test_y, score(:,2), 1);
figure, plot(xpos, ypos)
xlim([-0.05 1.05]), ylim([-0.05 1.05])
xlabel('\bf FP rate'), ylabel('\bf TP rate')
title('\bf ROC for Classification by SVM_class 1')


