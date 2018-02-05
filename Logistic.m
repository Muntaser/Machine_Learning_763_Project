%% import data
clear all
clc

%importing data
opts = detectImportOptions('HR_comma_sep.csv');
data = readtable('HR_comma_sep.csv',opts);

%remove test set w/ out replacement. 70/30 split
[test_data,idx] = datasample(data,450,'replace',false);
ind = [idx] ; % indices to be removed
data(ind, :) = [] ; % remove

%% data pre-processing 
%create dummy vars
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
avg_monthly_hours= z(:,2); %%average_montly_hours

%%class distrbution
s=summary(data);
employeeleft = find(data.left==1);
size(employeeleft);
employeenotleft = find(data.left==0);
size(employeenotleft);
unique(data.sales);

%normalize the data set
X=[ zlast_evaluation zsatisfaction_level  no_project years_spend avg_monthly_hours ...
 data.promotion_last_5years data.Work_accident dv_salary(:,1:2) dv_sales(:,1:9)];
Y= data.left;

%logistic regression
mdl = fitglm(X, Y, 'distr', 'binomial', 'link', 'logit')
p=mdl.Fitted.Response;
z=mdl.Fitted.LinearPredictor;
figure, gscatter (z, p, Y, 'br'); grid on 
%plotSlice (mdl)

%% confusion matrix, logistic 
ypredlr = double(predict(mdl, X) >= 0.5);
CFM = confusionmat(Y, ypredlr)
Accuracy = sum(diag(CFM))/sum(CFM(:))

disp("Class_1:Employee Left-Precision,Recall and F1 ")
Precision_C1= CFM(1,1)/sum(CFM(:,1))
Recall_C1 = CFM(1,1)/sum(CFM(1,:))
F1_C1=(2*Precision_C1*Recall_C1)/(Precision_C1+Recall_C1)

disp("Class_0: Employee Stayed-Precision,Recall and F1")
Precision_C0= CFM(2,2)/sum(CFM(:,2))
Recall_C0 = CFM(2,2)/sum(CFM(2,:))
F1_C0=(2*Precision_C0*Recall_C0)/(Precision_C0+Recall_C0)

%roc for class 1
probability1 = predict(mdl, X);
probability0 = 1-(probability1);
[xpos, ypos, T, AUC1] = perfcurve(Y, probability1, 1);
figure, plot(xpos, ypos)
xlim([-0.05 1.05]), ylim([-0.05 1.05])
xlabel('\bf FP rate'), ylabel('\bf TP rate')
title('\bf ROC for Classification wrt "Employee Left"')

%roc for class 0
[xpos, ypos, T, AUC0] = perfcurve(Y, probability0, 0);
figure, plot(xpos, ypos)
xlim([-0.05 1.05]), ylim([-0.05 1.05])
xlabel('\bf FP rate'), ylabel('\bf TP rate')
title('\bf ROC for Classification wrt "Employee Stayed"')
disp("AUC0 AUC1")
disp([AUC0 AUC1])

%% Remove Outliers 
%Find the observations with Cook's distance values that are relatively larger than the other observations 
%with Cook's distances exceeding the threshold value
cooksd = find((mdl.Diagnostics.CooksDistance)>3*mean(mdl.Diagnostics.CooksDistance)); %found 1,313, 9% of data, too many
outlier = find((mdl.Diagnostics.CooksDistance)>5*mean(mdl.Diagnostics.CooksDistance)); %used 5 as threshold, 628, ok
mdl = fitlm(X, Y, 'Exclude',outlier);
mdl.ObservationInfo(outlier,:);
%Re-standardization 
newX = zscore(X);
newY = data.left;

%% Fit a cross-validated sequence of models with lasso, and plot the result:
[B,FitInfo] = lassoglm(newX, newY,'binomial',...
'NumLambda',25,'CV', 10,'PredictorNames',....
{'Last_evaluation' ,'Satisfaction_level',  'No_project', 'Years_spend', 'Avg_mothly_hours', ...
 'Promotion-In-5yrs', 'Work_accident', 'Salary_low','Salary_High', ......
 'Sales','Sales_A/c','Sales_hr','Sales_Tec','Sales_Sup','Sales_IT','Sales_Pro','Sales_RandD','Sales_Mark',});
lassoPlot(B,FitInfo,'PlotType', 'Lambda', 'XScale', 'log');
xlabel('lambda');
ylabel('Theta');
legend('show');

%nonzero predictors
indx = FitInfo.Index1SE;
B0 = B(:,indx);
nonzeros = sum(B0 ~= 0);
%create reqularized model
cnst = FitInfo.Intercept(indx);
B1 = [cnst;B0];
prdec=glmval(B1,newX,'logit');
%Use identified predictors nonzero ones
predictors = find(B0); % indices of nonzero predictors
mdl1 = fitglm(newX,newY,'linear',...
    'Distribution','binomial','PredictorVars',predictors);

%%confusion matrix 
PredictedClasses = double(predict(mdl1, newX) >= 0.5);
CFM = confusionmat(newY, PredictedClasses)

Accuracy = sum(diag(CFM))/sum(CFM(:))
disp("Class_1:Employee Left-Precision,Recall and F1 ")
Precision_C1= CFM(1,1)/sum(CFM(:,1))
Recall_C1 = CFM(1,1)/sum(CFM(1,:))
F1_C1=(2*Precision_C1*Recall_C1)/(Precision_C1+Recall_C1)

disp("Class_0: Employee Stayed-Precision,Recall and F1")
Precision_C0= CFM(2,2)/sum(CFM(:,2))
Recall_C0 = CFM(2,2)/sum(CFM(2,:))
F1_C0=(2*Precision_C0*Recall_C0)/(Precision_C0+Recall_C0)

%roc for class 1
probability1 = predict(mdl, newX);
probability0 = 1-(probability1);
[xpos, ypos, T, AUC1] = perfcurve(newY, probability1, 1);
figure, plot(xpos, ypos)
xlim([-0.05 1.05]), ylim([-0.05 1.05])
xlabel('\bf FP rate'), ylabel('\bf TP rate')
title('\bf ROC for Classification wrt "Employee Left"')

%roc for class 0
[xpos, ypos, T, AUC0] = perfcurve(newY, probability0, 0);
figure, plot(xpos, ypos)
xlim([-0.05 1.05]), ylim([-0.05 1.05])
xlabel('\bf FP rate'), ylabel('\bf TP rate')
title('\bf ROC for Classification wrt "Employee Stayed"')
disp("AUC0 AUC1")
disp([AUC0 AUC1])
