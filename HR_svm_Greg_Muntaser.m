%HR_SVM
clear all
clc

%%importing data
%opts = detectImportOptions('C:\tmp\HR_comma_sep.csv');
%data = readtable('C:\tmp\HR_comma_sep.csv',opts);

%remove test set w/ out replacement. 70/30 split
%[test_data,idx] = datasample(data,4500,'replace',false);
%ind = [idx] ; % indices to be removed
%data(ind, :) = [] ; % remove

%create test set
%filename = 'C:\tmp\test_set_hr.xlsx';
%writetable(test_data,filename,'Sheet',1,'Range','A1')

%create training set
%filename2 = 'C:\tmp\training_set_hr.xlsx';
%writetable(data,filename2,'Sheet',1,'Range','A1')

clear all
clc

%%importing data
%test set
opts = detectImportOptions('C:\tmp\test_set_hr.xlsx');
test_data = readtable('C:\tmp\test_set_hr.xlsx',opts);
%training set
opts2 = detectImportOptions('C:\tmp\training_set_hr.xlsx');
data = readtable('C:\tmp\training_set_hr.xlsx',opts);


%%data pre-processing 
%%create dummy vars
data.sales = nominal(data.sales);
dv_sales= dummyvar(data.sales);
data.salary = nominal(data.salary);
dv_salary= dummyvar(data.salary);%% salary

%Zscore data
zlast_evaluation = zscore(data.last_evaluation);
zsatisfaction_level = zscore(data.satisfaction_level);

z=zscore([ data.time_spend_company data.average_montly_hours data.number_project]);
no_project= z(:,3);%%number_project
years_spend= z(:,1);%%time_spend_company
avg_mothly_hours= z(:,2); %%average_montly_hours

%%class distrbution
s=summary(data);
employeeleft = find(data.left==1);
size(employeeleft);
employeenotleft = find(data.left==0);
size(employeenotleft);
unique(data.sales);

%normalized data set
x=[ zlast_evaluation zsatisfaction_level  no_project years_spend avg_mothly_hours ...
 data.promotion_last_5years data.Work_accident dv_salary(:,1:2) dv_sales(:,1:9)];
y= data.left;


%SVM model
SVMModel = fitcsvm(x,y,'Standardize',true);


%test data set normalizeation
test_data.sales = nominal(test_data.sales);
dv_sales= dummyvar(test_data.sales);
test_data.salary = nominal(test_data.salary);
dv_salary= dummyvar(test_data.salary);%% salary
%Zscore data
zlast_evaluation = zscore(test_data.last_evaluation);
zsatisfaction_level = zscore(test_data.satisfaction_level);
z=zscore([ test_data.time_spend_company test_data.average_montly_hours test_data.number_project]);
no_project= z(:,3);%%number_project
years_spend= z(:,1);%%time_spend_company
avg_mothly_hours= z(:,2); %%average_montly_hours
%new x
x2=[ zlast_evaluation zsatisfaction_level  no_project years_spend avg_mothly_hours ...
 test_data.promotion_last_5years test_data.Work_accident dv_salary(:,1:2) dv_sales(:,1:9)];
y2=test_data.left;


%Build ROC curve with no crossval. Positive class curve
[label,score] = predict(SVMModel,x2);
[Xa,Ya,T,AUC] = perfcurve(y,score(:,[2]),1);
figure, plot(Xa,Ya)
title('Left. No crossval. Training data')
xlabel('False positive rate'); ylabel('True positive rate');
 
[Xa,Ya,T,AUC] = perfcurve(y,score(:,[1]),0);
figure, plot(Xa,Ya)
title('Stayed. No crossval. Training data')
xlabel('False positive rate'); ylabel('True positive rate');
 
%confusion matrix
%positive class
[label,score] = predict(SVMModel,x2);
CFM = confusionmat(y2,label)
accuracy = sum(diag(CFM))/sum(CFM(:))
precision = CFM(1,1) / (CFM(1,1) + CFM(2,1))
Recall = CFM(1,1) / (CFM(1,1) + CFM(1,2))
f1_one = (2*Recall*precision) / (Recall + precision)

%negative class
label(label==0) = 1.5;
label(label==1) = 0;
label(label==1.5) = 1;
CFM = confusionmat(y,label)
accuracy = sum(diag(CFM))/sum(CFM(:))
precision = CFM(1,1) / (CFM(1,1) + CFM(2,1))
Recall = CFM(1,1) / (CFM(1,1) + CFM(1,2)) 
f1_two = (2*Recall*precision) / (Recall + precision) 

%accuracy, precsion, recall for class 1
accuracy = sum(diag(CFM))/sum(CFM(:))
precision1= CFM(2,2)/sum(CFM(:,2))
recall1 = CFM(2,2)/sum(CFM(2,:))
f1=(2*precision*Recall)/(precision+Recall)







%average probability of getting classified as 1
ScoreSVMModel = fitPosterior(SVMModel,x,y);
[label,score] = predict(ScoreSVMModel,x2);  

one_class = score(:,2);
one = one_class(one_class >.5);
one = mean(one);%.6975

two_class = score(:,1);
two = two_class(two_class >.5);
two = mean(two);%.8184

%rerun model
SVMModel = fitcsvm(x,y,'Standardize',true);


 
%Model with crossval. 5 fold / 10 fold / 15 fold
SVMModel = fitcsvm(x,y,'Standardize',true,'Crossval','on');
SVMModel = fitcsvm(x,y,'Standardize',true,'Crossval','on','kfold',15);
 
%Build ROC curve with  crossval. Positive class curve

[label, score] = predict(SVMModel.Trained{2,1}, x2);

[Xa,Ya,T,AUC1] = perfcurve(y2,score(:,[1]),0);
figure, plot(Xa,Ya)
title('Stayed. crossval. test set')
xlabel('False positive rate'); ylabel('True positive rate');
 
[Xa,Ya,T,AUC2] = perfcurve(y2,score(:,[2]),1);
figure, plot(Xa,Ya)
title('Left. crossval. test set')
xlabel('False positive rate'); ylabel('True positive rate');

%confusion matrix
%positive class
CFM = confusionmat(y2,label)
accuracy = sum(diag(CFM))/sum(CFM(:))
precision = CFM(1,1) / (CFM(1,1) + CFM(2,1))
Recall = CFM(1,1) / (CFM(1,1) + CFM(1,2))
f1_one = (2*Recall*precision) / (Recall + precision)

%negative class
label(label==0) = 1.5;
label(label==1) = 0;
label(label==1.5) = 1 ;
CFM = confusionmat(y2,label)
accuracy = sum(diag(CFM))/sum(CFM(:))
precision = CFM(1,1) / (CFM(1,1) + CFM(2,1))
Recall = CFM(1,1) / (CFM(1,1) + CFM(1,2)) 
f1_two = (2*Recall*precision) / (Recall + precision) 
 
%run for each model
L = kfoldLoss(SVMModel);
 
 
 
%Model with box constraint adjustment
%Trying various box constraints
SVMModel = fitcsvm(x,y,'Standardize',true,'Crossval','on','kfold',15,'BoxConstraint',2e-1);
L = kfoldLoss(SVMModel);
SVMModel = fitcsvm(x,y,'Standardize',true,'Crossval','on','kfold',15,'BoxConstraint',1e-5);
L = kfoldLoss(SVMModel);
SVMModel = fitcsvm(x,y,'Standardize',true,'Crossval','on','kfold',15,'BoxConstraint',1e-1);
L = kfoldLoss(SVMModel);
SVMModel = fitcsvm(x,y,'Standardize',true,'Crossval','on','kfold',15,'BoxConstraint',1e5);
L = kfoldLoss(SVMModel);

%fingind the optimal box constraint and sigma
sigma = optimizableVariable('sigma',[1e-2,1e2],'Transform','log');
box = optimizableVariable('box',[1e-2,1e2],'Transform','log');

c = cvpartition(10499,'kfold',15);

minfn = @(z)kfoldLoss(fitcsvm(x,y,'CVPartition',c,...
'KernelFunction','rbf','BoxConstraint',z.box,...
'KernelScale',z.sigma));
    
results = bayesopt(minfn,[sigma,box],'IsObjectiveDeterministic',true,...
'AcquisitionFunctionName','expected-improvement-plus')    

%retrain model w/ best scores
z(1) = results.XAtMinObjective.sigma;
z(2) = results.XAtMinObjective.box;
SVMModel = fitcsvm(x,y,'KernelFunction','rbf',...
    'KernelScale',1.0396,'BoxConstraint',10.368);
 
[label,score] = predict(SVMModel,x2);
[Xa,Ya,T,AUC] = perfcurve(y2,score(:,[2]),1);
figure, plot(Xa,Ya)
title('Left. No crossval. Training data')
xlabel('False positive rate'); ylabel('True positive rate');
 
[Xa,Ya,T,AUC] = perfcurve(y2,score(:,[1]),0);
figure, plot(Xa,Ya)
title('Stayed. No crossval. Training data')
xlabel('False positive rate'); ylabel('True positive rate'); 
[label, score] = predict(SVMModel, x2);

%confusion matrix
%positive class
CFM = confusionmat(y2,label)
accuracy = sum(diag(CFM))/sum(CFM(:))
precision = CFM(1,1) / (CFM(1,1) + CFM(2,1))
Recall = CFM(1,1) / (CFM(1,1) + CFM(1,2))
f1_one = (2*Recall*precision) / (Recall + precision)

%negative class
label(label==0) = 1.5;
label(label==1) = 0;
label(label==1.5) = 1 ;
CFM = confusionmat(y2,label)
accuracy = sum(diag(CFM))/sum(CFM(:))
precision = CFM(1,1) / (CFM(1,1) + CFM(2,1))
Recall = CFM(1,1) / (CFM(1,1) + CFM(1,2)) 
f1_two = (2*Recall*precision) / (Recall + precision) 

%accuracy, precsion, recall for negative class
accuracy = sum(diag(CFM))/sum(CFM(:))
precision= CFM(2,2)/sum(CFM(:,2))
Recall = CFM(2,2)/sum(CFM(2,:))
f1=(2*precision*Recall)/(precision+Recall)



%average probability of getting classified as 1
ScoreSVMModel = fitPosterior(SVMModel,x,y);
[label,score] = predict(ScoreSVMModel,x2);  

one_class = score(:,2);
one = one_class(one_class >.5);
one = mean(one);%.6975

two_class = score(:,1);
two = two_class(two_class >.5);
two = mean(two);%.8184

%output final predictions
T = array2table(label);
filename = 'C:\tmp\final_response_predict.xlsx';
writetable(T,filename,'Sheet',1,'Range','A1')
 


%Support Vectors are too wide
%Adjust BoxConstraint
SVMModel = fitcsvm(x,y,'Standardize',true,'Crossval','on','kfold',15,'BoxConstraint',.045,'KernelFunction','Linear',...
'KernelScale',1/sqrt(.01));

SVMModel = fitcsvm(x,y,'Standardize',true,'Crossval','on','kfold',15,'BoxConstraint',.045);
[label, score] = predict(SVMModel.Trained{6,1}, x2);
mseLin = kfoldLoss(SVMModel)

SVMModel = fitcsvm(x,y,'Standardize',true,'Crossval','on','kfold',15,'BoxConstraint', 1,'KernelFunction','rbf');

%confusion matrix
%positive class
CFM = confusionmat(y2,label)
accuracy = sum(diag(CFM))/sum(CFM(:))
precision = CFM(1,1) / (CFM(1,1) + CFM(2,1))
Recall = CFM(1,1) / (CFM(1,1) + CFM(1,2))
f1_one = (2*Recall*precision) / (Recall + precision)

%negative class
label(label==0) = 1.5;
label(label==1) = 0;
label(label==1.5) = 1 ;
CFM = confusionmat(y2,label)
accuracy = sum(diag(CFM))/sum(CFM(:))
precision = CFM(1,1) / (CFM(1,1) + CFM(2,1))
Recall = CFM(1,1) / (CFM(1,1) + CFM(1,2)) 
f1_two = (2*Recall*precision) / (Recall + precision)  
 
%feature selection based on Beta (weight)
%No improvement / discard
new_data=x(:,[1 4 5]);
new_test=x2(:,[1 4 5]);
SVMModel = fitcsvm(new_data,y,'Standardize',true,'Crossval','on','kfold',15,'BoxConstraint', 1,'KernelFunction','rbf');
[label, score] = predict(SVMModel.Trained{4,1}, new_test);
mseLin = kfoldLoss(SVMModel)

%Get labels and scores. Scores are distance from D.B.
[label,score] = predict(SVMModel,x2);
 
ScoreSVMModel = fitPosterior(SVMModel,x,y);
[label,score] = predict(ScoreSVMModel,x2); 
 

sv = SVMModel.Trained{4,1}.SupportVectors;
figure
gscatter(x(:,4),x(:,5),y)
hold on
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10)
legend('stayed','left','Support Vector')
hold off








