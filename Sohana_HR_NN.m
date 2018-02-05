

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

%NaN value
idx=isnan(X)| isinf(X);
for idx = 1:length(X)
 n = X(idx);
 n = rand(length(idx), 1);
end


%  transpose data
Y(Y~=0) =  1;
Y  =  Y';
Y  =  [~Y; Y];
X  =X';


% Solve a Pattern Recognition Problem with a Neural Network
%   X - input data.
%   Y - target data.

x = X;
t = Y;

% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network

hiddenLayerSize = [30,20,10];
net = patternnet(hiddenLayerSize, trainFcn);



% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
net.trainParam.epochs=197;
% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y);
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);

%Change layers
transferFcn=net.layers;

net.layers{2}.transferFcn='tansig';
net.layers{3}.transferFcn='poslin'; % ReLU layer before 'softmax';
net.layers{4}.transferFcn = 'softmax';

Yhat =net(x);
yperdiciton  = vec2ind(Yhat);
yperdiciton(yperdiciton==1)=0;
yperdiciton(yperdiciton==2)=1;
yperdiciton= yperdiciton';
% View the Network
view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
figure, plotconfusion(t,y)
%figure, plotroc(t,y)

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
