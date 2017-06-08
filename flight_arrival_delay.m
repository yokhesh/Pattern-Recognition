% BINARY CLASSIFICATION FOR THE FLIGHT ARRIVAL DELAY%
%Only 20000 datas are used in training both the binary classifier and the regression model
%Another 10000 datas are used in testing the binary classifier and the regression model
tic
A = xlsread('data_check.xlsx');
[m,n] = size(A);
g = zeros(m,1);
% Creating the output labels for all the datas
%output labels of training data used in training the classifier
%output label of testing data used in determining the accuracy of the classifier
for i = 1:m
    if A(i,1) ~= 0
        g(i,1) = 1;
    end
end
x1 = 1;x2 = 1;
for i = 1:m
    if g(i,1) == 0
        no_delay(x1,:) = A(i,2:end);
        x1 = x1+1;
    elseif g(i,1) == 1
        delay(x2,:) = A(i,2:end);
        x2 = x2+1;
    end
end
[p,q] = size(no_delay);
[u,i] = size(delay);
tr_req = (p/3)*2;tr_req1 = (u/3)*2;
g_delay = ones*ones(u,1);
g_no_delay = zeros(p,1);
training = [no_delay(1:tr_req,:);delay(1:tr_req1,:)];
testing = [no_delay(tr_req+1:end,:);delay(tr_req1+1:end,:)];
training_labels = [g_no_delay(1:tr_req,:);g_delay(1:tr_req1,:)];
testing_labels = [g_no_delay(tr_req+1:end,:);g_delay(tr_req1+1:end,:)];

%Extracting statistical features from training and testing data sets

%statistical training feature
mean_train = mean(training');
std_train = std(training');
ran_train = range(training');
var_train = var(training');
kurt_train = kurtosis(training');
skew_train = skewness(training');
training = [mean_train;std_train;ran_train;kurt_train;skew_train]';

%statistical test feature
mean_test = mean(testing');
std_test = std(testing');
ran_test = range(testing');
var_test = var(testing');
kurt_test = kurtosis(testing');
skew_test = skewness(testing');
testing = [mean_test;std_test;ran_test;kurt_test;skew_test]';

%discriminant function used for building the classifier

%mean and covariance of training data belonging class 1
mean1 = mean(training(1:tr_req,:));
covariance1 = cov(training(1:tr_req,:));
%mean and covariance of training data belonging class 2
mean2 = mean(training(tr_req+1:end,:));
covariance2 = cov(training(tr_req+1:end,:));
lo = size(testing,1);

%Probability density function of test data for class w1 will be stored in g1
g1 = ones(lo,1);
for j = 1:lo
ki = inv(covariance1)*(testing(j,:) - mean1)';
g1(j,1) = (-0.5)*(testing(j,:) - mean1)*ki - (0.5*log(det(covariance1)));
end

% Probability density function of test data for class w2 will be stored in g2
g2 = ones(lo,1);
for j = 1:lo
ki = (covariance2)\ (testing(j,:) - mean2)';
g2(j,1) = (-0.5)*(testing(j,:) - mean2)*ki  - (0.5*log(det(covariance2)));
end
 
classification = zeros(lo,1);
%Binary classification of 0 indicates no delay 
%Binary Classification of 1 indicates delay
for i = 1:lo
    if g1(i,1)>g2(i,1)
        classification(i,1) = 0;
    else
        classification(i,1) = 1;
    end
end
 con = confusionmat(classification,testing_labels);
 pr1 = con(1,1)+con(2,2);
 disp('Accuracy for the Binary classifier');
 accuracy = pr1/length(testing);
 disp(accuracy*100);
 disp('The classification made by the binary classifier on the test data can be found in the variable named "classification"');
 disp('The original correct classification for the test data can be found in the variable named "testing_labels"');
 fprintf('\n');
% REGRESSION MODEL PREDICTING THE EXTENT OF THE DELAY%

% Inputs taken are  CRS arrival time and the original arrival time
% The arrival delay time is obtained from the excel file only to check the accuracy of the regression model
A = xlsread('regression_ch.xlsx');
trai_dat = (size(A,1)/3)*2;

%getting the training data
training_reg = A(1:trai_dat,2:end);
training_reg1 = training_reg(:,1);training_reg2 = training_reg(:,2);
minutes1 = mod(training_reg1, 100);minutes2 = mod(training_reg2, 100);    
hour1 = training_reg1./100;hour2 = training_reg2./100;
hour1 = fix(hour1);hour2 = fix(hour2);
htm = hour1.*60;htm1 = hour2.*60;
training_reg1 = htm+minutes1;training_reg2 = htm1+minutes2;
%Final training data after pre-processing
training_reg = [training_reg1 training_reg2];

%getting the testing data
testing_reg = A(trai_dat+1:end,2:end);
testing_reg1 = testing_reg(:,1);testing_reg2 = testing_reg(:,2);
minutes1t = mod(testing_reg1, 100);minutes2t = mod(testing_reg2, 100);    
hour1t = testing_reg1./100;hour2t = testing_reg2./100;
hour1t = fix(hour1t);hour2t = fix(hour2t);
htmt = hour1t.*60;htm1t = hour2t.*60;
testing_reg1 = htmt+minutes1t;testing_reg2 = htm1t+minutes2t;
%Final testing data after pre-processing
testing_reg = [testing_reg1 testing_reg2];

%creating training and testing output labels
%output labels of training data used in training the regression model
%output label of testing data used in determining the accuracy of the regression model
tr_labels = training_reg(:,2) - training_reg(:,1);
te_labels = testing_reg(:,2)-testing_reg(:,1);
y = [tr_labels;te_labels];
y1 = A(:,1);y1 = y1(20001:end,:);
training_labels_reg = y(1:trai_dat,:);
testing_labels_reg = y(trai_dat+1:end,:);
m = length(training_labels_reg);
n = length(testing_labels_reg);

%scaling the training and testing data for better accuracy
training_reg = training_reg./10000;testing_reg = testing_reg./10000;

%Adding the bias column in the training and testing data
training_reg = [ones(m,1), training_reg(:,:)];
testing_reg = [ones(n,1), testing_reg(:,:)];

% Initial declaration of theta, alpha 
theta = zeros(3, 1);
alpha = 1;
cost = 100;count = 0;

% This while will execute until the cost becomes less than 0.01
% the less the cost of the model, the better the accuracy

while (cost > 0.8)
    count = count+1;
    %finding hypothesis
    hypo = training_reg*theta;
    
    %finding cost
    difference = sum((hypo - training_labels_reg).^2);
    diff = difference;
    cost = diff./(2*m);
    
    %gradient descent
    %for theta0
    difference1 = sum(hypo - training_labels_reg);
    second_t0 = (difference1*alpha)/m;
    theta0 = theta(1,1) - second_t0;
    %for theta1
    second_t1 = (hypo - training_labels_reg)'*training_reg(:,2);
    second_t11 = (second_t1*alpha)/m;
    theta1 = theta(2,1) - second_t11;
    %for theta2
    second_t2 = (hypo - training_labels_reg)'*training_reg(:,3);
    second_t22 = (second_t2*alpha)/m;
    theta2 = theta(3,1) - second_t22;
    
    theta = [theta0;theta1;theta2];
end

% Using the theta obtained from linear regression to predict the output for the test data

%Negative value indicates the number of minutes the flight  will arrive early
%For example: A value of -5 in prediction indicates the flight will arrive 5 minutes in advance to scheduled arrival time

%Positive value indicates the delay in flight arrival in minutes
%For example: A value of 5 in prediction indicates the flight will be delayed by 5 minutes  
    prediction = testing_reg*theta;
% The variable prediction contains the flight arrival delay prediction for all the test data
   prediction = round(prediction,0);
    
    c2 = 0;
    for i = 1:length(prediction)
        if prediction(i,1) == y1(i,1)
            c2 = c2+1;
        end
    end
    accuracy_reg = c2/size(y1,1);
     disp('Accuracy for the Regression model');
     disp(accuracy_reg*100 );
     disp('The predicted flight delay value for the test data can be found in the variable named "prediction"');
    disp('The original correct flight delay values for the test data can be found in the variable named "y1"');
     time_taken = toc;