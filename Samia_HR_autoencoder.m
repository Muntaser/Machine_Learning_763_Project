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
% Train a softmax layer for classification using the |features| .
softnet = trainSoftmaxLayer(feat2,y,'MaxEpochs',400);
% Stack the encoder and the softmax layer to form a deep network.
deepnet = stack(autoenc1,autoenc2,softnet);
% View the stacked network.
view(deepnet);
% Perform fine tuning
deepnet = train(deepnet,x,y);
%  view the results using a confusion matrix.
yh = deepnet(test_x);
yperdiciton  = vec2ind(yh)
yperdiciton(yperdiciton==1)=0;
yperdiciton(yperdiciton==2)=1;
% Test the Network
plotconfusion(test_y,yh)
[MaxValue PredictedClasses] = max(yh);
[MaxValue TrueClasses] =  max(test_y);
C_auto = confusionmat(TrueClasses,  PredictedClasses)
P1   =  C_auto(1,1) /  sum(C_auto(:,1));
R1   =  C_auto(1,1) /  sum(C_auto(1,:));
F1   =  (2*P1*R1) /  (P1+R1);
P2   =  C_auto(2,2) /  sum(C_auto(:,2));
R2   =  C_auto(2,2) /  sum(C_auto(2,:));
F2   =  (2*P2*R2) /  (P2+R2);disp([P1 R1   F1   P2   R2   F2])
%roc 
plotroc(test_y,yh)
% visulize the train data, feature1 & 2
% colormap(copper) % You realize this affects final image (H)?
subplot(2,2,1)
% Now make an RGB image that matches display from IMAGESC:
C = colormap;  % Get the figure's colormap.
L = size(C,1);
% Scale the matrix to the range of the map.
xs = round(interp1(linspace(min(x(:)),max(x(:)),L),1:L,x));
im = ind2rgb(xs, colormap);
image(im)
title('trained data')
test_xs = round(interp1(linspace(min(test_x(:)),max(test_x(:)),L),1:L,test_x));
imtx = ind2rgb(test_xs, colormap);
subplot(2,2,2)
image(imtx)  % Does this image match the other one?
title('test data')
%feat1 and feat2
% colormap(copper) % You realize this affects final image (H)?
subplot(2,2,3)
f1s = round(interp1(linspace(min(feat1(:)),max(feat1(:)),L),1:L,feat1));
imf1 = ind2rgb(f1s, colormap);
image(imf1)
title(' feature 1')
f2s = round(interp1(linspace(min(feat2(:)),max(feat2(:)),L),1:L,feat2));
imf2 = ind2rgb(f2s, colormap);
subplot(2,2,4)
image(imf2)  % Does this image match the other one?
title('feature 2')



