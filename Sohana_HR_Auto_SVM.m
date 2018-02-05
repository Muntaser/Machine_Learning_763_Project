clear all
clc
%%importing data
opts = detectImportOptions('HR_comma_sep.csv');     % set import options
opts.VariableTypes(9:10) = {'categorical'};         % turn text to categorical
data = readtable('HR_comma_sep.csv',opts);

%%data pre-processing 
%%create dummy vars
data.sales = nominal(data.sales);
dv_sales= dummyvar(data.sales);
dx_sales=dv_sales(:,1:9);
idx = find(isnan(dx_sales));
dx_sales(idx) = rand(length(idx), 1);
data.salary = nominal(data.salary);
dv_salary= dummyvar(data.salary);%% salary
dx_salary=dv_salary(:,1:2);


%Zscore data
last_evaluation = zscore(data.last_evaluation);
satisfaction_level = zscore(data.satisfaction_level);
no_project= zscore(data.number_project);%%number_project
years_spend= zscore(data.time_spend_company);%%time_spend_company
avg_mothly_hours= zscore(data.average_montly_hours); %%average_montly_hours
work_accident= zscore(data.Work_accident);
promotion_last_5years=zscore(data.promotion_last_5years);



%%class distrbution
s=summary(data);
employeeleft = find(data.left==1);
size(employeeleft);
employeenotleft = find(data.left==0);
size(employeenotleft);
unique(data.sales);


%normalized data set
X=[satisfaction_level last_evaluation  no_project  avg_mothly_hours years_spend...
  work_accident promotion_last_5years dx_sales dx_salary ];
Y= data.left;


%test_data processing
opts = detectImportOptions('test_set_hr.xlsx');
test_data = readtable('test_set_hr.xlsx',opts);
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
XTest=[ zlast_evaluation zsatisfaction_level  no_project years_spend avg_mothly_hours test_data.promotion_last_5years test_data.Work_accident dv_salary(:,1:2) dv_sales(:,1:9)];
YTest=test_data.left;

%training_data processing
opts = detectImportOptions('training_set_hr.xlsx');
training_data = readtable('training_set_hr.xlsx',opts);
%training data set normalizeation
training_data.sales = nominal(training_data.sales);
dv_sales= dummyvar(training_data.sales);
training_data.salary = nominal(training_data.salary);
dv_salary= dummyvar(training_data.salary);%% salary
zlast_evaluation = zscore(training_data.last_evaluation);
zsatisfaction_level = zscore(training_data.satisfaction_level);
z=zscore([ training_data.time_spend_company training_data.average_montly_hours training_data.number_project]);
no_project= z(:,3);%%number_project
years_spend= z(:,1);%%time_spend_company
avg_mothly_hours= z(:,2); %%average_montly_hours
%new x & y
XTrain =[ zlast_evaluation zsatisfaction_level  no_project years_spend avg_mothly_hours training_data.promotion_last_5years training_data.Work_accident dv_salary(:,1:2) dv_sales(:,1:9)];
YTrain=training_data.left;


%  transpose data
Y(Y~=0) =  1;
Y  =  Y';
Y  =  [~Y; Y];
X  =X';

YTrain (YTrain ~=0)=1;
YTrain =YTrain';
YTrain =[~YTrain; YTrain ];
XTrain=XTrain';

YTest (YTest~=0) =1;
YTest=YTest';
YTest =[~YTest; YTest];
XTest = XTest';

%autoencoder layer1
hiddenSize = 15;
autoenc1 = trainAutoencoder(XTrain, hiddenSize, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization', 0.001, ...
    'SparsityRegularization', 4, ...
    'SparsityProportion', 0.05, ...
    'DecoderTransferFunction','purelin');

% Extract the features from hidden layer1.
feat1= encode (autoenc1, XTrain);
view(autoenc1);

%autoencoder layer2
hiddenSize = 10;
autoenc2 = trainAutoencoder(feat1,hiddenSize,...
     'MaxEpochs',300, ...
    'L2WeightRegularization',0.001,...
    'SparsityRegularization',4,...
    'SparsityProportion',0.05,...
    'DecoderTransferFunction','purelin');

% Extract the features from hidden layer2.
feat2= encode (autoenc2, feat1);
view (autoenc2);


%autoencoder layer3
hiddenSize = 7;
autoenc3 = trainAutoencoder(feat2,hiddenSize,...
     'MaxEpochs',200, ...
    'L2WeightRegularization',0.001,...
    'SparsityRegularization',4,...
    'SparsityProportion',0.05,...
    'DecoderTransferFunction','purelin');

% Extract the features from hidden layer3.
feat3= encode (autoenc3, feat2);
view(autoenc3);
% Train a softmax layer for classification 
softnet = trainSoftmaxLayer(feat3, YTrain, 'MaxEpochs', 200);
view(softnet)
% Stack the encoder and the softmax layer to form a deep network.
deepnet = stack(autoenc1, autoenc2, autoenc3, softnet);


%Re-standardization 
znewX =zscore(feat3);
new_Y   = reshape(YTrain,10499,2);
newY = new_Y(:,1);


SVMModel = fitcsvm(znewX', newY', 'KernelFunction', 'rbf', 'Crossval', 'on');
[label,Score] = kfoldPredict(SVMModel);
CFM= confusionmat( newY',label)

%%confusion matrix for class 1 (left)
Accuracy = sum(diag(CFM))/sum(CFM(:))
disp("Class_1:Employee Left-Precision,Recall and F ")
Precision_C1= CFM(1,1)/sum(CFM(:,1))
Recall_C1 = CFM(1,1)/sum(CFM(1,:))
F_C1=(2*Precision_C1*Recall_C1)/(Precision_C1+Recall_C1)

disp("Class_0: Employee Stayed-Precision,Recall and F")
Precision_C0= CFM(2,2)/sum(CFM(:,2))
Recall_C0 = CFM(2,2)/sum(CFM(2,:))
F_C0=(2*Precision_C0*Recall_C0)/(Precision_C0+Recall_C0)

[xpos, ypos, T, AUC] = perfcurve(newY', Score(:,2), 1);
figure, plot(xpos, ypos) % plot ROC 
xlim([-0.05 1.05]), ylim([-0.05 1.05])
xlabel('\bf FP rate'), ylabel('\bf TP rate')
title('\bf ROC for Classification wrt "Employee Left"')

[xpos, ypos, T, AUC] = perfcurve(newY', Score(:,1), 0);
figure, plot(xpos, ypos) % plot ROC
xlim([-0.05 1.05]), ylim([-0.05 1.05])
xlabel('\bf FP rate'), ylabel('\bf TP rate')
title('\bf ROC for Classification wrt ""Employee Stayed"')




