%%%%%%%%%%%%%%%%%%%FIRST PART%%%%%%%%%%%%%%%%%%%%%%%
tic
disp('--------------FIRST PART-----------------')
 x = load ('train_sp2017_v19');
 y = load ('test_sp2017_v19');
 yt = y';
z = load ('ykrishn-classified-takehome1.txt');
correct_test = ones(15000,1);
m = 1;correct_train = ones(15000,1);
for j = 1:5000
    correct_train (j,1) = 1; si = j+5000;
    correct_train (si,1) = 2; ti = j+10000;
    correct_train (ti,1) = 3;
end
% Creating a vector with the correct classification for test dat
for i = 1:2500
    i = m;
    correct_test(i,1) = 3;correct_test(i+1,1) = 1;correct_test(i+2,1) = 2;correct_test(i+3,1) = 3;correct_test(i+4,1) = 2;correct_test(i+5,1) = 1;
    m = i+6;
end
% Constructing the confusion matrix
fcm = zeros(3,3);
for j = 1:15000
    firstindex = correct_test(j,1) ;  secondindex = z(j,1) ;
    fcm(firstindex,secondindex) = fcm(firstindex,secondindex)+1;
end
confusion_matrix_part1 = fcm
% FInding out the probability of error
correct_values1 = fcm(1,1)+fcm(2,2)+fcm(3,3);
wrong_value1 = 15000 - correct_values1;
probability_of_error_using_confusion_matrix_part1 = wrong_value1/15000
disp('---------------SECOND PART-----------------')
%%%%%%%%%%%%%%%%%%%%%%%SECOND PART%%%%%%%%%%%%%%%%%%%%%%%%%
x1 = ones(5000,4);x2 = ones(5000,4);x3 = ones(5000,4);
 for i = 1:5000
    x1(i,1:4) = x(i,1:4);
     k = 5000+i;
    x2(i,1:4) = x(k,1:4);
     k1 = 10000+i;
    x3(i,1:4) = x(k1,1:4);
 end
 %Generating hyperplane equation separating class 1 and class 2
 Atrain1 = [x1 ones(5000,1); -x2  -ones(5000,1)];
 b_next1 = ones(10000,1); w1 = ones(5,1); alpha= 0.7;count1 = 0;check1 = 1;
 while (check1 > 0)
     check1 = 0;
     b1 = b_next1;
     w1 = (inv(Atrain1'*Atrain1))*Atrain1'*b1;
     e1 = ((Atrain1*w1)-b1);
     b_next1 = b1 + (alpha.*((e1+abs(e1))/2));
     count1 = count1+1;
     for i = 1:10000
         if e1(i,1) > 0.0001
             check1 = check1+1;
         end
     end
 end
 pre1 = [x1 ones(5000,1); x2  ones(5000,1)];
 sc1 = pre1*w1; sp1 = zeros(10000,1);
 for i =1:10000
 if sc1(i,1) > 0 
     sp1(i,1) = 1;
 else
     sp1(i,1) = 2;
 end
 end
 fcmsp1 = zeros(2,2);
for j = 1:10000
    firstindex = correct_train(j,1) ;  secondindex = sp1(j,1) ;
    fcmsp1(firstindex,secondindex) = fcmsp1(firstindex,secondindex)+1;
end
disp('Given below is the hyperplane classification performance for each pair of classes') 
confusion_matrix_using_h12_classifying_class1_and_class2 = fcmsp1
probability_of_error_h12 = ((10000-(fcmsp1(1,1)+fcmsp1(2,2)))/10000)
 %Generating hyperplane equation separating class 2 and class 3
 Atrain2 = [x2 ones(5000,1); -x3  -ones(5000,1)];
 b_next2 = ones(10000,1); w2 = ones(5,1); alpha= 0.7;count2 = 0;check2 = 1;
 while (check2 > 0)
     check2 = 0;
     b2 = b_next2;
     w2 = (inv(Atrain2'*Atrain2))*Atrain2'*b2;
     e2 = ((Atrain2*w2)-b2);
     b_next2 = b2 + (alpha.*((e2+abs(e2))/2));
     count2 = count2+1;
     for i = 1:10000
         if e2(i,1) > 0.0001
             check2 = check2+1;
         end
     end
 end
  pre2 = [x2 ones(5000,1); x3  ones(5000,1)];
 sc2 = pre2*w2; sp2 = zeros(10000,1);
 for i =1:10000
 if sc2(i,1) > 0 
     sp2(i,1) = 2;   
 else
     sp2(i,1) = 3;
 end
 end
  fcmsp2 = zeros(2,2);
for j = 1:10000
    p = j+5000;
    firstindex = correct_train(p,1) ;  secondindex = sp2(j,1) ;
    firstindex(firstindex==2) = 1;firstindex(firstindex==3) = 2;
    secondindex(secondindex==2) = 1;secondindex(secondindex==3) = 2;
    fcmsp2(firstindex,secondindex) = fcmsp2(firstindex,secondindex)+1;
end
confusion_matrix_using_h23_classifying_class2_and_class3 = fcmsp2
probability_of_error_h23 = ((10000-(fcmsp2(1,1)+fcmsp2(2,2)))/10000)
%Generating hyperplane equation separating class 3 and class 1
 Atrain3 = [x3 ones(5000,1); -x1  -ones(5000,1)];
 b_next3 = ones(10000,1); w3 = ones(5,1); alpha= 0.7;count3 = 0;check3 = 1;
 while (check3 > 0)
     check3 = 0;
     b3 = b_next3;
     w3 = (inv(Atrain3'*Atrain3))*Atrain3'*b3;
     e3 = ((Atrain3*w3)-b3);
     b_next3 = b3 + (alpha.*((e3+abs(e3))/2));
     count3 = count3+1;
     for i = 1:10000
         if e3(i,1) > 0.0001
             check3 = check3+1;
         end
     end
 end
  pre3 = [x3 ones(5000,1); x1  ones(5000,1)];
 sc3 = pre3*w3; sp3 = zeros(10000,1);
 for i =1:10000
 if sc3(i,1) > 0 
     sp3(i,1) = 3;
 else
     sp3(i,1) = 1;
 end
 end
 correct_train3 = correct_train(10001:15000,1);correct_train1 = correct_train(1:5000,1);
  fcmsp3 = zeros(2,2);p = 5000;correct_train31 = [correct_train3;correct_train1];
for j = 1:10000
    firstindex = correct_train31(j,1) ;  secondindex = sp3(j,1) ;
    firstindex(firstindex==1) = 2;firstindex(firstindex==3) = 1;
    secondindex(secondindex==1) = 2;secondindex(secondindex==3) = 1;
    fcmsp3(firstindex,secondindex) = fcmsp3(firstindex,secondindex)+1;
end
confusion_matrix_using_h31_classifying_class3_and_class1 = fcmsp3
probability_of_error_h31 = ((10000-(fcmsp3(1,1)+fcmsp3(2,2)))/10000)
 %Using the hyperplane equation to determine the classification of training data
 appendtr = [x ones(15000,1)];
 h12tr = appendtr*w1;
 h23tr = appendtr*w2;
 h31tr = appendtr*w3;p2classification_tr = zeros(15000,1);
 for i = 1:15000
     if h12tr(i,1) > 0 && h31tr(i,1) < 0 
         p2classification_tr(i,1) = 1;
     elseif h23tr(i,1) > 0 && h12tr(i,1) < 0
         p2classification_tr(i,1) = 2;
     else
         p2classification_tr(i,1) = 3;
     end
 end
 %Using the hyperplane equation to determine the classification of testing data
 append = [y ones(15000,1)];
 h12 = append*w1;
 h23 = append*w2;
 h31 = append*w3;p2classification = zeros(15000,1);
 for i = 1:15000
     if h12(i,1) > 0 && h31(i,1) < 0 
         p2classification(i,1) = 1;
     elseif h23(i,1) > 0 && h12(i,1) < 0
         p2classification(i,1) = 2;
     else
         p2classification(i,1) = 3;
     end
 end
% Constructing the confusion matrix for classification made on training and testing data using the hyperplane equation
% Probability of error for classification made on training and testing data using the hyperplane equation
disp('Confusion matrix for the classification of testing and training data using all three hyperplane equation')
fcm2 = zeros(3,3);
first_in2 = correct_test; second_in2 = p2classification; nam = 'test';
for i = 1:2
fcm2 = zeros(3,3);
    for j = 1:15000
    firstindex = first_in2(j,1) ;  secondindex = second_in2(j,1) ;
    fcm2(firstindex,secondindex) = fcm2(firstindex,secondindex)+1;
    end
    fprintf('The confusion matrix for %s data using Ho-Kayshap \n',nam); disp(fcm2);
    correct_values1 = fcm2(1,1)+fcm2(2,2)+fcm2(3,3);
    wrong_value1 = 15000 - correct_values1;
    probability_of_error_for_part2 = wrong_value1/15000;
    fprintf('The probability of error for %s data using Ho-Kayshap \n',nam); disp(probability_of_error_for_part2);
    first_in2 = correct_train; second_in2 = p2classification_tr;
    nam = 'training';
end
disp('------------------THIRD PART------------------')
%%%%%%%%%%%%%%%%%%%%%%%%%THIRD PART%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Only a smaller area around the test vector is searched for training examples.
% If the required training examples is not found, then the area in which the search is done will be increased.
 ate = y(:,1);bte = y(:,2);cte = y(:,3);dte = y(:,4);
 atr = x(:,1);btr = x(:,2);ctr = x(:,3);dtr = x(:,4);tni = 0;
 h = 0;ge1 = 8;ge2 = 5;ge3 = 8;ge4 = 8; num=1;
 p3classification = zeros(15000,3);count = 0;
 classification1 =zeros(15000,5);classification2 =zeros(15000,5);classification3 =zeros(15000,5);
    for i = num:15000 
        l = 0; h =0;me(i) = 0; dist = 0;p = 0;
        while me(i) < 5
         for j = 1:15000
%Defining the limit of the first feature vector in which the search is conducted
             gadd1 = ate(i) + ge1; gsub1 = ate(i) - ge1;
%Selecting the training examples that lies within the limit of the first feature
             if atr(j) <= gadd1 && atr(j) >= gsub1
%Defining the limit of the second feature vector in which the search is conducted
                 gadd2 = bte(i) + ge2; gsub2 = bte(i) - ge2;
%Selecting the training examples that lies within the limit of the first two features
                       if btr(j) <= gadd2 && btr(j) >= gsub2
%Defining the limit of the third feature vector in which the search is conducted
                           gadd3 = cte(i) + ge3; gsub3 = cte(i) - ge3;
%Selecting the training examples that lies within the limit of the first three features
                           if ctr(j) <= gadd3 && ctr(j) >= gsub3
%Defining the limit of the fourth feature vector in which the search is conducted
                               gadd4 = dte(i) + ge4; gsub4 = dte(i) - ge4;
%Selecting the training examples that lies within the limit of the first three features
                               if dtr(j) <= gadd4 && dtr(j) >= gsub4
% Finding the distance between the test vector and training vectors that are within the particular area
                                   count = count+1;
                                   h = h+1;me(i) = me(i) +1; 
                                   l = horzcat(atr(j),btr(j),ctr(j),dtr(j));
                                   tester = horzcat(ate(i),bte(i),cte(i),dte(i));
                                   dist(h,1) = sqrt(((l(1,1)-tester(1,1))^2) + ((l(1,2)-tester(1,2))^2) + ((l(1,3)-tester(1,3))^2) + ((l(1,4)-tester(1,4))^2));
                                   p(h,1) = j;
                               end
                           end
                       end
             end
             
         end
% Checking if the required number of distance samples has been calculated from the training data
         if me(i) < 5     
             ge1 = ge1+8;ge2 = ge2 + 5;ge3 = ge3+8;ge4 = ge4+8;
         elseif me(i) >= 5
             ivalue(i) = i;
             ge1 = 8;ge2 =  5;ge3 = 8;ge4 = 8;
              final = horzcat(p,dist);
             min_dist = sortrows(final,2);tni=tni+1;
% 1-NNR classification
             if min_dist(1,1) <= 5000 
                 p3classification(i,1) = 1; 
             elseif  min_dist(1,1) >5000 && min_dist(1,1) <= 10000
                 p3classification(i,1) = 2;
             elseif min_dist(1,1)>10000
                 p3classification(i,1) = 3; 
             end
% 3-NNR classification
classification3_1 = 0;classification3_2 = 0;classification3_3 = 0;             
             for k3 = 1:3
                 if min_dist(k3,1) <= 5000 
                     classification3_1 = classification3_1 +1;
                 elseif min_dist(k3,1) >5000 && min_dist(k3,1) <= 10000
                     classification3_2 = classification3_2 +1;
                 elseif min_dist(k3,1)>10000 
                     classification3_3 = classification3_3 +1;
                 end
             end
             if classification3_1 >= classification3_2
                 if classification3_1 >= classification3_3
                     p3classification(i,2) = 1;
                 elseif classification3_3 > classification3_1
                     p3classification(i,2) = 3;
                 end
             elseif classification3_2 > classification3_1
                 if classification3_2 >= classification3_3
                     p3classification(i,2) = 2;
                 elseif classification3_3 > classification3_2
                     p3classification(i,2) = 3;
                 end
             end
% 5-NNR classification
classification5_1 = 0;classification5_2 = 0;classification5_3 = 0;
             for k5 = 1:5
                 if min_dist(k5,1) <= 5000 
                     classification5_1 = classification5_1 +1;
                 elseif min_dist(k5,1) >5000 && min_dist(k5,1) <= 10000
                     classification5_2 = classification5_2 +1;
                 elseif min_dist(k5,1)>10000 
                     classification5_3 = classification5_3 +1;
                 end
             end
             if classification5_1 >= classification5_2
                 if classification5_1 >= classification5_3
                     p3classification(i,3) = 1;
                 elseif classification5_3 > classification5_1
                     p3classification(i,3) = 3;
                 end
             elseif classification5_2 > classification5_1
                 if classification5_2 >= classification5_3
                     p3classification(i,3) = 2;
                 elseif classification5_3 > classification5_2
                     p3classification(i,3) = 3;
                 end
             end         
         end
        end
    end
   te = 1;
% Calculating the confusion matrix and probability of error for 1-NNR, 3-NNR and 5-NNR
for i = 1:3 
     fcm3 = zeros(3,3);
    for j = 1:15000
    firstindex = correct_test(j,1) ;  secondindex = p3classification(j,i) ;
    fcm3(firstindex,secondindex) = fcm3(firstindex,secondindex)+1;
    end
confusion_matrix_part3 = fcm3;
fprintf('The confusion matrix for %d-NNR \n',te);
disp(confusion_matrix_part3); 
correct_values3 = fcm3(1,1)+fcm3(2,2)+fcm3(3,3);
wrong_value3 = 15000 - correct_values3;
fprintf('The Probability of error for %d-NNR \n',te);
te = te+2;
probability_of_error_using_confusion_matrix_part3 = wrong_value3/15000;
disp(probability_of_error_using_confusion_matrix_part3); 
end
    disp('-------------------FOURTH PART-------------------')
%%%%%%%%%%%%%%%%%%%FOURTH PART%%%%%%%%%%%%%%%%%%%%%%
M = mean(x);
x_norm = ones(15000,4);
% Mean normalization 
for i = 1:15000
    x_norm(i,:) = x(i,:) - M(:,:);
end
covariance = cov(x_norm);
[e_vector, e_val] = eig(covariance);
%Finding out the reduced training and testing data.
newtraining_set = x*e_vector(:,3:4);
newtest_set = y*e_vector(:,3:4); 
xt = newtraining_set';
xpca1 = ones(2,5000);
xpca2 = ones(2,5000);
xpca3 = ones(2,5000);
for i = 1:5000
    xpca1(:,i) = xt(:,i);
     k = 5000+i;
    xpca2(:,i) = xt(:,k);
     k1 = 10000+i;
    xpca3(:,i) = xt(:,k1);
end
mean1 = mean(xpca1');
covariance1 = cov(xpca1');
  mean2 = mean(xpca2');
covariance2 = cov(xpca2');
   mean3 = mean(xpca3');
covariance3 = cov(xpca3');
% Classification made for the reduced testing data
g1 = ones(1,15000);g2 = ones(1,15000);g3 = ones(1,15000);
for t = 1:15000
g1(1,t) = (-0.5)*(newtest_set(t,:) - mean1)* inv(covariance1) *(newtest_set(t,:) - mean1).' - (2*log(2*pi))- (0.5*log(det(covariance1)));
g2(1,t) = (-0.5)*(newtest_set(t,:) - mean2)* inv(covariance2)  *(newtest_set(t,:) - mean2).' - (2*log(2*pi))- (0.5*log(det(covariance2)));
g3(1,t) = (-0.5)*(newtest_set(t,:) - mean3)* inv(covariance3) *(newtest_set(t,:) - mean3).' - (2*log(2*pi))- (0.5*log(det(covariance3)));
end
p4classification = zeros(15000,1);
for an = 1:15000
    if (g1(1,an) > g2(1,an))
        if (g1(1,an) > g3(1,an))
            p4classification(an,1) = 1;
        elseif (g3(1,an) > g1(1,an))
            p4classification(an,1) = 3;
        end
    elseif (g2(1,an) > g1(1,an))
         if (g2(1,an) > g3(1,an))
            p4classification(an,1) = 2;
        elseif (g3(1,an) > g2(1,an))
            p4classification(an,1) = 3;
         end
    end
end
% Classification made for the reduced training data
g1tr = ones(1,15000);g2tr = ones(1,15000);g3tr = ones(1,15000);
for t = 1:15000
g1tr(1,t) = (-0.5)*(newtraining_set(t,:) - mean1)* inv(covariance1) *(newtraining_set(t,:) - mean1).' - (2*log(2*pi))- (0.5*log(det(covariance1)));
g2tr(1,t) = (-0.5)*(newtraining_set(t,:) - mean2)* inv(covariance2)  *(newtraining_set(t,:) - mean2).' - (2*log(2*pi))- (0.5*log(det(covariance2)));
g3tr(1,t) = (-0.5)*(newtraining_set(t,:) - mean3)* inv(covariance3) *(newtraining_set(t,:) - mean3).' - (2*log(2*pi))- (0.5*log(det(covariance3)));
end
p4classification_tr = zeros(15000,1);
for an = 1:15000
    if (g1tr(1,an) > g2tr(1,an))
        if (g1tr(1,an) > g3tr(1,an))
            p4classification_tr(an,1) = 1;
        elseif (g3tr(1,an) > g1tr(1,an))
            p4classification_tr(an,1) = 3;
        end
    elseif (g2tr(1,an) > g1tr(1,an))
         if (g2tr(1,an) > g3tr(1,an))
            p4classification_tr(an,1) = 2;
        elseif (g3tr(1,an) > g2tr(1,an))
            p4classification_tr(an,1) = 3;
         end
    end
end
first_in = correct_test; second_in = p4classification; nam = 'test';probability_of_error_for_part4 = 0;
for i = 1:2
fcm4t = zeros(3,3);
test =  probability_of_error_for_part4;
    for j = 1:15000
    firstindex = first_in(j,1) ;  secondindex = second_in(j,1) ;
    fcm4t(firstindex,secondindex) = fcm4t(firstindex,secondindex)+1;
    end
    fprintf('The confusion matrix for %s data using PCA \n',nam); disp(fcm4t);
    correct_values1 = fcm4t(1,1)+fcm4t(2,2)+fcm4t(3,3);
    wrong_value1 = 15000 - correct_values1;
    probability_of_error_for_part4 = wrong_value1/15000;
    fprintf('The probability of error for %s data using PCA \n',nam); disp(probability_of_error_for_part4);
    first_in = correct_train; second_in = p4classification_tr;
    nam = 'training';
end
dif = test - probability_of_error_using_confusion_matrix_part1;
fprintf('The increase in the probability of error due to dimension reduction using PCA is equal to %d \n',dif);
chang =  test/probability_of_error_using_confusion_matrix_part1;
fprintf('Therefore, the error in PCA has increased by %d \n',chang);
timeelapsed = toc
                              

                           
                           
