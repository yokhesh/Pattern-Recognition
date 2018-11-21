% Loading the Training and Testing data
x = load ('train_sp2017_v19');
y = load ('test_sp2017_v19');
correct_classification = 0;
wrong_classification = 0;
p_w1 = (1/3);
p_w2 = (1/3);
p_w3 = (1/3);
y = y.';
xt = x';
n1 = 0;
n2 = 0;
wrong_w1 = 0;
wrong_w2 = 0;
wrong_w3 = 0;
% Calculating the mean and Covariance matrix for class w1
x1 = ones(5000,4);
for i = 1:5000
    x1(i,1:4) = x(i,1:4);
end
mean1 = mean(x1);
mean1 = mean1.';
covariance1 = cov(x1);
covariance1 = covariance1.';
% Calculating the mean and Covariance matrix for class w2
x2 = ones(5000,4);
for i = 1:5000
    k = 5000+i;
    x2(i,1:4) = x(k,1:4);
end
mean2 = mean(x2);
mean2 = mean2.';
covariance2 = cov(x2);
covariance2 = covariance2.';
% Calculating the mean and Covariance matrix for class w3
x3 = ones(5000,4);
for i = 1:5000
    k = 10000+i;
    x3(i,1:4) = x(k,1:4);
end
mean3 = mean(x3);
mean3 = mean3.';
covariance3 = cov(x3);
covariance3 = covariance3.';



%classifying the test data
%Finding the probability density function of test data for class w1
%Probability density function of test data for class w1 will be stored in g1
g1 = ones(1,15000);
for j = 1:15000
g1(1,j) = (-0.5)*(y(1:4,j) - mean1).'* inv(covariance1)* (y(1:4,j) - mean1) - (2*log(2*pi))- (0.5*log(det(covariance1)));
end
% Prove that the testing and training data is gaussian
% In addition to this, the histogram is also used to test the data
% The following code for Lilliefors test is made as a comment as it slows down the operation considerably

% for tg = 1:15000
% X = xt((1:4),tg);
% h = lillietest(X);   
% if h == 0
%    n1 = n1+1;
% end
% Y = y((1:4),tg);
% h = lillietest(Y);   
% if h == 0
%    n2 = n2+1;
% end
% end
% Number_of_vectors_in_training_data_that_are_from_gaussian = n1;
% Number_of_vectors_in_testing_data_that_are_from_gaussian = n2;

%Finding the probability density function of test data for class w2
%Probability density function of test data for class w2 will be stored in g2
g2 = ones(1,15000);
for j = 1:15000
g2(1,j) = (-0.5)*(y(1:4,j) - mean2).'* inv(covariance2)* (y(1:4,j) - mean2) - (2*log(2*pi))- (0.5*log(det(covariance2)));
end
%Finding the probability density function of test data for class w3
%Probability density function of test data for class w3 will be stored in g3
g3 = ones(1,15000);
for j = 1:15000
g3(1,j) = (-0.5)*(y(1:4,j) - mean3).'* inv(covariance3)* (y(1:4,j) - mean3) - (2*log(2*pi))- (0.5*log(det(covariance3)));
end
% Comparing the probability density function of test data for Class w1,w1,w3 
for an = 1:15000
if (g1(1,an) > g2(1,an))
    if (g1(1,an) > g3(1,an))
        disp('1')
    elseif (g1(1,an) < g3(1,an))
        disp('3')
    end
elseif (g2(1,an) > g1(1,an))
    if (g2(1,an) > g3(1,an))
        disp('2')
    elseif (g2(1,an) < g3(1,an))
        disp('3')
    end
elseif (g3(1,an) > g1(1,an))
     if (g3(1,an) > g2(1,an))
        disp('3')
    elseif (g3(1,an) < g2(1,an))
        disp('2')
     end
end
end
%Finding the probability error
%Use the classifier on the training data
%Compare the classification obtained from classifier with the correct classification to get the probability error

%Using the classifier to find out the probability density of training data for class w1 

g1tr = ones(1,15000);
for t = 1:15000
g1tr(1,t) = (-0.5)*(xt(1:4,t) - mean1).'* inv(covariance1)* (xt(1:4,t) - mean1) - (2*log(2*pi))- (0.5*log(det(covariance1)));
end

%Using the classifier to find out the probability density of training data for class w2 
g2tr = ones(1,15000);
for t = 1:15000
g2tr(1,t) = (-0.5)*(xt(1:4,t) - mean2).'* inv(covariance2)* (xt(1:4,t) - mean2) - (2*log(2*pi))- (0.5*log(det(covariance2)));
end

%Using the classifier to find out the probability density of training data for class w3 
g3tr = ones(1,15000);
for t = 1:15000
g3tr(1,t) = (-0.5)*(xt(1:4,t) - mean3).'* inv(covariance3)* (xt(1:4,t) - mean3) - (2*log(2*pi))- (0.5*log(det(covariance3)));
end

% Comparing the obtained classification for the first 5000 samples (1 to 5000 samples) with its original classification
for p = 1:5000
    if g1tr(1,p) > g2tr(1,p) && g1tr(1,p) > g3tr(1,p)
        correct_classification = correct_classification + 1;
    elseif g1tr(1,p)<g2tr(1,p) 
        wrong_classification = wrong_classification + 1;
        wrong_w2 = wrong_w2+1;
    elseif g1tr(1,p) < g3tr(1,p)
        wrong_classification = wrong_classification + 1;
        wrong_w3 = wrong_w3+1;
    end
end
a = wrong_classification;
Number_of_vector_classified_under_w1_for_first_5000_samples = correct_classification;
Number_of_vector_classified_under_w2_for_first_5000_samples = wrong_w2;
Number_of_vector_classified_under_w3_for_first_5000_samples = wrong_w3;
cc1 = correct_classification;
wc1 = wrong_w2+wrong_w3;
% Comparing the obtained classification for the second 5000 samples (5001 to 10000 samples) with its original classification
wrong_w1 = 0;
wrong_w2 = 0;
wrong_w3 = 0;
correct_classification = 0;
for p = 5001:10000
    if g2tr(1,p) > g1tr(1,p) && g2tr(1,p) > g3tr(1,p)
        correct_classification = correct_classification + 1;
    elseif g2tr(1,p)<g1tr(1,p)  
        wrong_classification = wrong_classification + 1;
        wrong_w1 = wrong_w1+1;
    elseif g2tr(1,p) < g3tr(1,p)
        wrong_classification = wrong_classification + 1;
         wrong_w3 = wrong_w3+1;
    end
end
b = wrong_classification - a;
Number_of_vector_classified_under_w1_for_second_5000_samples = wrong_w1;
Number_of_vector_classified_under_w2_for_second_5000_samples = correct_classification;
Number_of_vector_classified_under_w3_for_second_5000_samples = wrong_w3;
cc2 = correct_classification;
wc2 = wrong_w1 + wrong_w3;
% Comparing the obtained classification for the third 5000 samples (10001 to 15000 samples) with its original classification
wrong_w1 = 0;
wrong_w2 = 0;
wrong_w3 = 0;
correct_classification = 0;
for p = 10001:15000
    if g3tr(1,p) > g2tr(1,p) && g3tr(1,p) > g1tr(1,p)
        correct_classification = correct_classification + 1;
    elseif g3tr(1,p)<g2tr(1,p)  
        wrong_classification = wrong_classification + 1;
        wrong_w2 = wrong_w2+1;
    elseif g3tr(1,p) < g1tr(1,p)
        wrong_classification = wrong_classification + 1;
        wrong_w1 = wrong_w1+1;
    end
end
c = wrong_classification - a - b;
Number_of_vector_classified_under_w1_for_third_5000_samples = wrong_w1;
Number_of_vector_classified_under_w2_for_third_5000_samples = wrong_w2;
Number_of_vector_classified_under_w3_for_third_5000_samples = correct_classification;
cc3 = correct_classification;
wc3 = wrong_w1 + wrong_w2;
Total_number_of_training_vector_classified_correctly = cc1+cc2+cc3;
Total_number_of_training_vector_classified_wrongly = wc1+wc2+wc3;
Probability_of_error = (wrong_classification)/15000
    
    
