%%%%%%%%%%%%%%%%%%%%%%C2MEANS%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
disp('C-Means');
disp('USING EUCLIDEAN DISTANCE');
x = load ('train_sp2017_v19');
y = load ('test_sp2017_v19');
c1 = x(1:5000,:);
c2 = x(5001:10000,:);
c3 = x(10001:15000,:);
m1 = mean(c1); m2 = mean(c2);m3 = mean(c3);
c = 2;
mean_training = mean(x);
di1 = zeros(15000,2);
for i = 1:15000
    di1(i,1) = norm(mean_training - x(i,:));
    di1(i,2) = i;
end
max_dist1 = sortrows(di1,1);
mu1 = x(max_dist1(15000,2),:);
di2 = zeros(15000,2);
for i = 1:15000
    di2(i,1) = norm(mu1 - x(i,:));
    di2(i,2) = i;
end
max_dist2 = sortrows(di2,1);
mu2 = x(max_dist2(15000,2),:);un_mu1 = mu1;un_mu2 = mu2;
p=0;iteration = 0;
while (p==0)
    count = 0;iteration = iteration+1;dis = zeros(c,15000);
    dis(1,:) = pdist2(mu1,x);
    dis(2,:) = pdist2(mu2,x);
    [value,index] = min(dis);
    n = sum(index(:) == 1); m = sum(index(:) == 2); 
    c2_euc_cluster1 = zeros(n,4);c2_euc_cluster2 = zeros(m,4);
    ni = 1;mi = 1;
    for i = 1:15000
        if dis(1,i) <= dis(2,i)
            c2_euc_cluster1(ni,:) = x(i,:);
            ni = ni+1;
        elseif dis(1,i) > dis(2,i)
            c2_euc_cluster2(mi,:) = x(i,:);
            mi = mi+1;
        end
    end
mu11 = mean(c2_euc_cluster1);
mu21 = mean(c2_euc_cluster2);
if (mu1==mu11 & mu2 == mu21)
    p = 1;
else
    p = 0;
    mu1 = mu11;mu2 = mu21;
end
end
disp(' For C=2');
disp('Size of Cluster 1');disp(size(c2_euc_cluster1,1));
disp('Size of Cluster 2');disp(size(c2_euc_cluster2,1));
M = [mu1;mu2];
c2_dis_euc1 = pdist2(M,m1);c2_dis_euc2 = pdist2(M,m2);
c2_dis_euc3 = pdist2(M,m3);
%%%%%%%%%%%%%%%%%%%%%%C3MEANS%%%%%%%%%%%%%%%%%%%%%%%%%%%
c = 3;
mu1 = un_mu1;mu2 = un_mu2;
mean_det = (mu1+mu2)./2;
di3 = zeros(15000,2);
for i = 1:15000
    di3(i,1) = norm(mean_det - x(i,:));
    di3(i,2) = i;
end
max_dist3 = sortrows(di3,1);
mu3 = x(max_dist3(1,2),:);
p=0;iteration = 0;un_mu3 = mu3;
while (p==0)
    count = 0;iteration = iteration+1;dis = zeros(c,15000);
    dis(1,:) = pdist2(mu1,x);
    dis(2,:) = pdist2(mu2,x);
    dis(3,:) = pdist2(mu3,x);
    [value,index] = min(dis);
    n = sum(index(:) == 1); m = sum(index(:) == 2); o = sum(index(:) == 3);
    c3_euc_cluster1 = zeros(n,4);c3_euc_cluster2 = zeros(m,4);c3_euc_cluster3 = zeros(o,4);
    ni = 1;mi = 1;oi = 1;
for i = 1:15000
    if dis(1,i) <= dis(2,i) && dis(1,i) <= dis(3,i)
        c3_euc_cluster1(ni,:) = x(i,:);
        ni = ni+1;
    elseif dis(2,i) <= dis(1,i) && dis(2,i) <= dis(3,i)
        c3_euc_cluster2(mi,:) = x(i,:);
        mi = mi+1;
    elseif dis(3,i) <= dis(2,i) && dis(3,i) <= dis(1,i)
        c3_euc_cluster3(oi,:) = x(i,:);
        oi = oi+1;
    end
end
mu11 = mean(c3_euc_cluster1);
mu21 = mean(c3_euc_cluster2);
mu31 = mean(c3_euc_cluster3);
if (mu1==mu11 & mu2 == mu21 & mu3 == mu31)
    p = 1;
else
    p = 0;
    mu1 = mu11;mu2 = mu21;mu3 = mu31;
end
end
disp(' For C=3');
disp('Size of Cluster 1');disp(size(c3_euc_cluster1,1));
disp('Size of Cluster 2');disp(size(c3_euc_cluster2,1));
disp('Size of Cluster 3');disp(size(c3_euc_cluster3,1));
M = [mu1;mu2;mu3];
c3_dis_euc1 = pdist2(M,m1);c3_dis_euc2 = pdist2(M,m2);
c3_dis_euc3 = pdist2(M,m3);
%%%%%%%%%%%%%%%%%%%%%%C4MEANS%%%%%%%%%%%%%%%%%%%%%%%%%%%
c = 4;
mu1 = un_mu1;mu2 = un_mu2;mu3 = un_mu3;

mean_det2 = (mu1+mu3)./2;mean_det3 = (mu2+mu3)./2;
di4 = zeros(15000,2);di5 = zeros(15000,2);
for i = 1:15000
    di4(i,1) = norm(mean_det2 - x(i,:));di5(i,1) = norm(mean_det3 - x(i,:));
    di4(i,2) = i;di5(i,2) = i;
end
min_dist4 = sortrows(di4,1);min_dist5 = sortrows(di5,1);
mu4 = x(min_dist4(1,2),:);mu5 = x(min_dist5(1,2),:);
un_mu4 = mu4;un_mu5 = mu5;

p=0;iteration = 0;
while (p==0)
    count = 0;iteration = iteration+1;dis = zeros(c,15000);
    dis(1,:) = pdist2(mu1,x);
    dis(2,:) = pdist2(mu2,x);
    dis(3,:) = pdist2(mu4,x);
    dis(4,:) = pdist2(mu5,x);
    [value,index] = min(dis);
    n = sum(index(:) == 1); m = sum(index(:) == 2); o = sum(index(:) == 3);q = sum(index(:) == 4);
    c4_euc_cluster1 = zeros(n,4);c4_euc_cluster2 = zeros(m,4);c4_euc_cluster3 = zeros(o,4);c4_euc_cluster4 = zeros(q,4);
    ni = 1;mi = 1;oi = 1;qi = 1;
     for i = 1:15000
        if dis(1,i) <= dis(2,i) && dis(1,i) <= dis(3,i) && dis(1,i) <= dis(4,i) 
            c4_euc_cluster1(ni,:) = x(i,:);
            ni = ni+1;
        elseif dis(2,i) <= dis(1,i) && dis(2,i) <= dis(3,i) && dis(2,i) <= dis(4,i) 
            c4_euc_cluster2(mi,:) = x(i,:);
            mi = mi+1;
        elseif dis(3,i) <= dis(2,i) && dis(3,i) <= dis(1,i) && dis(3,i) <= dis(4,i) 
            c4_euc_cluster3(oi,:) = x(i,:);
            oi = oi+1;
        elseif dis(4,i) <= dis(2,i) && dis(4,i) <= dis(1,i) && dis(4,i) <= dis(3,i) 
            c4_euc_cluster4(qi,:) = x(i,:);
            qi = qi+1;
        end
    end
mu11 = mean(c4_euc_cluster1);
mu21 = mean(c4_euc_cluster2);
mu31 = mean(c4_euc_cluster3);
mu41 = mean(c4_euc_cluster4);
if (mu1==mu11 & mu2 == mu21 & mu4 == mu31 & mu5 == mu41)
    p = 1;
else
    p = 0;
    mu1 = mu11;mu2 = mu21;mu4 = mu31;mu5 = mu41;
end
end
disp(' For C=4');
disp('Size of Cluster 1');disp(size(c4_euc_cluster1,1));
disp('Size of Cluster 2');disp(size(c4_euc_cluster2,1));
disp('Size of Cluster 3');disp(size(c4_euc_cluster3,1));
disp('Size of Cluster 4');disp(size(c4_euc_cluster4,1));
M = [mu1;mu2;mu3;mu4];
c4_dis_euc1 = pdist2(M,m1);c4_dis_euc2 = pdist2(M,m2);
c4_dis_euc3 = pdist2(M,m3);
%%%%%%%%%%%%%%%%%%%%%%C5MEANS%%%%%%%%%%%%%%%%%%%%%%%%%%%
c = 5;
mu1 = un_mu1;mu2 = un_mu2;mu3 = un_mu3;
mu4 = un_mu4;mu5 = un_mu5;
p=0;iteration = 0;
while (p==0)
    count = 0;iteration = iteration+1;dis = zeros(c,15000);
    dis(1,:) = pdist2(mu1,x);
    dis(2,:) = pdist2(mu2,x);
    dis(3,:) = pdist2(mu3,x);
    dis(4,:) = pdist2(mu4,x);
    dis(5,:) = pdist2(mu5,x);
    [value,index] = min(dis);
    n = sum(index(:) == 1); m = sum(index(:) == 2); o = sum(index(:) == 3);
    q = sum(index(:) == 4);u = sum(index(:) == 5);
    c5_euc_cluster1 = zeros(n,4);c5_euc_cluster2 = zeros(m,4);c5_euc_cluster3 = zeros(o,4);
    c5_euc_cluster4 = zeros(q,4);c5_euc_cluster5 = zeros(u,4);
    ni = 1;mi = 1;oi = 1;qi = 1;ui = 1;
    for i = 1:15000
        if dis(1,i) <= dis(2,i) && dis(1,i) <= dis(3,i) && dis(1,i) <= dis(4,i) && dis(1,i) <= dis(5,i)
            c5_euc_cluster1(ni,:) = x(i,:);
            ni = ni+1;
        elseif dis(2,i) <= dis(1,i) && dis(2,i) <= dis(3,i) && dis(2,i) <= dis(4,i) && dis(2,i) <= dis(5,i)
            c5_euc_cluster2(mi,:) = x(i,:);
            mi = mi+1;
        elseif dis(3,i) <= dis(2,i) && dis(3,i) <= dis(1,i) && dis(3,i) <= dis(4,i) && dis(3,i) <= dis(5,i)
            c5_euc_cluster3(oi,:) = x(i,:);
            oi = oi+1;
        elseif dis(4,i) <= dis(2,i) && dis(4,i) <= dis(1,i) && dis(4,i) <= dis(3,i) && dis(4,i) <= dis(5,i)
            c5_euc_cluster4(qi,:) = x(i,:);
            qi = qi+1;
        elseif dis(5,i) <= dis(2,i) && dis(5,i) <= dis(1,i) && dis(5,i) <= dis(3,i) && dis(5,i) <= dis(4,i)
            c5_euc_cluster5(ui,:) = x(i,:);
            ui = ui+1;
        end
    end
mu11 = mean(c5_euc_cluster1);
mu21 = mean(c5_euc_cluster2);
mu31 = mean(c5_euc_cluster3);
mu41 = mean(c5_euc_cluster4);
mu51 = mean(c5_euc_cluster5);
if (mu1 == mu11 & mu2 == mu21 & mu3 == mu31 & mu4 == mu41 & mu5 == mu51)
    p = 1;
else
    p = 0;
    mu1 = mu11;mu2 = mu21;mu3 = mu31;mu4 = mu41;mu5 = mu51;
end
end
disp(' For C=5');
disp('Size of Cluster 1');disp(size(c5_euc_cluster1,1));
disp('Size of Cluster 2');disp(size(c5_euc_cluster2,1));
disp('Size of Cluster 3');disp(size(c5_euc_cluster3,1));
disp('Size of Cluster 4');disp(size(c5_euc_cluster4,1));
disp('Size of Cluster 5');disp(size(c5_euc_cluster5,1));
M = [mu1;mu2;mu3;mu4;mu5];
c5_dis_euc1 = pdist2(M,m1);c5_dis_euc2 = pdist2(M,m2);
c5_dis_euc3 = pdist2(M,m3);
%%%%%%%%%%%%%%%%%%%%%%MANHATTAN_C2MEANS%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('USING MANHATTAN DISTANCE');
p=0;iteration = 0;mu1 = un_mu1;mu2 = un_mu2;
while (p==0)
    count = 0;iteration = iteration+1;dis = zeros(2,15000);
    P = abs(bsxfun(@minus,x,mu1));
    dis(1,:) = sum(P');
    Q = abs(bsxfun(@minus,x,mu2));
    dis(2,:) = sum(Q');
    [value,index] = min(dis);
    n = sum(index(:) == 1); m = sum(index(:) == 2); 
    c2_man_cluster1 = zeros(n,4);c2_man_cluster2 = zeros(m,4);
    ni = 1;mi = 1;
    for i = 1:15000
        if dis(1,i) <= dis(2,i)
            c2_man_cluster1(ni,1:4) = x(i,:);
            ni = ni+1;
        elseif dis(1,i) > dis(2,i)
            c2_man_cluster2(mi,1:4) = x(i,:);
            mi = mi+1;
        end
    end
mu11 = mean(c2_man_cluster1);
mu21 = mean(c2_man_cluster2);
if (mu1==mu11 & mu2 == mu21)
    p = 1;
else
    p = 0;
    mu1 = mu11;mu2 = mu21;
end
end
disp(' For C=2');
disp('Size of Cluster 1');disp(size(c2_man_cluster1,1));
disp('Size of Cluster 2');disp(size(c2_man_cluster2,1));
M = [mu1;mu2];
c2_dis_man1 = pdist2(M,m1);c2_dis_man2 = pdist2(M,m2);
c2_dis_man3 = pdist2(M,m3);
%%%%%%%%%%%%%%%%%%%%%%MANHATTAN_C3MEANS%%%%%%%%%%%%%%%%%%%%%%%%%%%
p=0;iteration = 0;mu1 = un_mu1;mu2 = un_mu2;mu3 = un_mu3;
while (p==0)
    count = 0;iteration = iteration+1;dis = zeros(3,15000);
    P = abs(bsxfun(@minus,x,mu1));
    dis(1,:) = sum(P');
    Q = abs(bsxfun(@minus,x,mu2));
    dis(2,:) = sum(Q');
    R = abs(bsxfun(@minus,x,mu3));
    dis(3,:) = sum(R');
    [value,index] = min(dis);
    n = sum(index(:) == 1); m = sum(index(:) == 2); o = sum(index(:) == 3);
    c3_man_cluster1 = zeros(n,4);c3_man_cluster2 = zeros(m,4);c3_man_cluster3 = zeros(o,4);
    ni = 1;mi = 1;oi = 1;
for i = 1:15000
    if dis(1,i) <= dis(2,i) && dis(1,i) <= dis(3,i)
        c3_man_cluster1(ni,1:4) = x(i,:);
        ni = ni+1;
    elseif dis(2,i) <= dis(1,i) && dis(2,i) <= dis(3,i)
        c3_man_cluster2(mi,1:4) = x(i,:);
        mi = mi+1;
    elseif dis(3,i) <= dis(2,i) && dis(3,i) <= dis(1,i)
        c3_man_cluster3(oi,1:4) = x(i,:);
        oi = oi+1;
    end
end
mu11 = mean(c3_man_cluster1);
mu21 = mean(c3_man_cluster2);
mu31 = mean(c3_man_cluster3);
if (mu1==mu11 & mu2 == mu21 & mu3 == mu31)
    p = 1;
else
    p = 0;
    mu1 = mu11;mu2 = mu21;mu3 = mu31;
end
end
disp(' For C=3');
disp('Size of Cluster 1');disp(size(c3_man_cluster1,1));
disp('Size of Cluster 2');disp(size(c3_man_cluster2,1));
disp('Size of Cluster 3');disp(size(c3_man_cluster3,1));
M = [mu1;mu2;mu3];
c3_dis_man1 = pdist2(M,m1);c3_dis_man2 = pdist2(M,m2);
c3_dis_man3 = pdist2(M,m3);
%%%%%%%%%%%%%%%%%%%%%%MANHATTAN_C4MEANS%%%%%%%%%%%%%%%%%%%%%%%%%%%
p=0;iteration = 0;mu1 = un_mu1;mu2 = un_mu2;mu3 = un_mu3;mu4 = un_mu4;mu5 = un_mu5;
while (p==0)
    count = 0;iteration = iteration+1;dis = zeros(4,15000);
    P = abs(bsxfun(@minus,x,mu1));
    dis(1,:) = sum(P');
    Q = abs(bsxfun(@minus,x,mu2));
    dis(2,:) = sum(Q');
    R = abs(bsxfun(@minus,x,mu3));
    dis(3,:) = sum(R');
    S = abs(bsxfun(@minus,x,mu5));
    dis(4,:) = sum(S');
    [value,index] = min(dis);
    n = sum(index(:) == 1); m = sum(index(:) == 2); o = sum(index(:) == 3);q = sum(index(:) == 4);
    c4_man_cluster1 = zeros(n,4);c4_man_cluster2 = zeros(m,4);c4_man_cluster3 = zeros(o,4);c4_man_cluster4 = zeros(q,4);
    ni = 1;mi = 1;oi = 1;qi = 1;
     for i = 1:15000
        if dis(1,i) <= dis(2,i) && dis(1,i) <= dis(3,i) && dis(1,i) <= dis(4,i) 
            c4_man_cluster1(ni,1:4) = x(i,:);
            ni = ni+1;
        elseif dis(2,i) <= dis(1,i) && dis(2,i) <= dis(3,i) && dis(2,i) <= dis(4,i) 
            c4_man_cluster2(mi,1:4) = x(i,:);
            mi = mi+1;
        elseif dis(3,i) <= dis(2,i) && dis(3,i) <= dis(1,i) && dis(3,i) <= dis(4,i) 
            c4_man_cluster3(oi,1:4) = x(i,:);
            oi = oi+1;
        elseif dis(4,i) <= dis(2,i) && dis(4,i) <= dis(1,i) && dis(4,i) <= dis(3,i) 
            c4_man_cluster4(qi,1:4) = x(i,:);
            qi = qi+1;
        end
    end
mu11 = mean(c4_man_cluster1);
mu21 = mean(c4_man_cluster2);
mu31 = mean(c4_man_cluster3);
mu41 = mean(c4_man_cluster4);
if (mu1==mu11 & mu2 == mu21 & mu4 == mu31 & mu5 == mu41)
    p = 1;
else
    p = 0;
    mu1 = mu11;mu2 = mu21;mu4 = mu31;mu5 = mu41;
end
end
disp(' For C=4');
disp('Size of Cluster 1');disp(size(c4_man_cluster1,1));
disp('Size of Cluster 2');disp(size(c4_man_cluster2,1));
disp('Size of Cluster 3');disp(size(c4_man_cluster3,1));
disp('Size of Cluster 4');disp(size(c4_man_cluster4,1));
M = [mu1;mu2;mu3;mu4];
c4_dis_man1 = pdist2(M,m1);c4_dis_man2 = pdist2(M,m2);
c4_dis_man3 = pdist2(M,m3);
%%%%%%%%%%%%%%%%%%%%%%MANHATTAN_C5MEANS%%%%%%%%%%%%%%%%%%%%%%%%%%%
p=0;iteration = 0;mu1 = un_mu1;mu2 = un_mu2;mu3 = un_mu3;mu4 = un_mu4;mu5 = un_mu5;
while (p==0)
    count = 0;iteration = iteration+1;dis = zeros(5,15000);
    P = abs(bsxfun(@minus,x,mu1));
    dis(1,:) = sum(P');
    Q = abs(bsxfun(@minus,x,mu2));
    dis(2,:) = sum(Q');
    R = abs(bsxfun(@minus,x,mu3));
    dis(3,:) = sum(R');
    S = abs(bsxfun(@minus,x,mu4));
    dis(4,:) = sum(S');
    T = abs(bsxfun(@minus,x,mu5));
    dis(5,:) = sum(T');
    [value,index] = min(dis);
    n = sum(index(:) == 1); m = sum(index(:) == 2); o = sum(index(:) == 3);
    q = sum(index(:) == 4);u = sum(index(:) == 5);
    c5_man_cluster1 = zeros(n,4);c5_man_cluster2 = zeros(m,4);c5_man_cluster3 = zeros(o,4);
    c5_man_cluster4 = zeros(q,4);c5_man_cluster5 = zeros(u,4);
    ni = 1;mi = 1;oi = 1;qi = 1;ui = 1;
    for i = 1:15000
        if dis(1,i) <= dis(2,i) && dis(1,i) <= dis(3,i) && dis(1,i) <= dis(4,i) && dis(1,i) <= dis(5,i)
            c5_man_cluster1(ni,1:4) = x(i,:);
            ni = ni+1;
        elseif dis(2,i) <= dis(1,i) && dis(2,i) <= dis(3,i) && dis(2,i) <= dis(4,i) && dis(2,i) <= dis(5,i)
            c5_man_cluster2(mi,1:4) = x(i,:);
            mi = mi+1;
        elseif dis(3,i) <= dis(2,i) && dis(3,i) <= dis(1,i) && dis(3,i) <= dis(4,i) && dis(3,i) <= dis(5,i)
            c5_man_cluster3(oi,1:4) = x(i,:);
            oi = oi+1;
        elseif dis(4,i) <= dis(2,i) && dis(4,i) <= dis(1,i) && dis(4,i) <= dis(3,i) && dis(4,i) <= dis(5,i)
            c5_man_cluster4(qi,1:4) = x(i,:);
            qi = qi+1;
        elseif dis(5,i) <= dis(2,i) && dis(5,i) <= dis(1,i) && dis(5,i) <= dis(3,i) && dis(5,i) <= dis(4,i)
            c5_man_cluster5(ui,1:4) = x(i,:);
            ui = ui+1;
        end
    end
mu11 = mean(c5_man_cluster1);
mu21 = mean(c5_man_cluster2);
mu31 = mean(c5_man_cluster3);
mu41 = mean(c5_man_cluster4);
mu51 = mean(c5_man_cluster5);
if (mu1==mu11 & mu2 == mu21 & mu3 == mu31 & mu4 == mu41 & mu5 == mu51)
    p = 1;
else
    p = 0;
    mu1 = mu11;mu2 = mu21;mu3 = mu31;mu4 = mu41;mu5 = mu51;
end
end
disp(' For C=5');
disp('Size of Cluster 1');disp(size(c5_man_cluster1,1));
disp('Size of Cluster 2');disp(size(c5_man_cluster2,1));
disp('Size of Cluster 3');disp(size(c5_man_cluster3,1));
disp('Size of Cluster 4');disp(size(c5_man_cluster4,1));
disp('Size of Cluster 5');disp(size(c5_man_cluster5,1));
M = [mu1;mu2;mu3;mu4;mu5];
c5_dis_man1 = pdist2(M,m1);c5_dis_man2 = pdist2(M,m2);
c5_dis_man3 = pdist2(M,m3);
%%%%%%%%%%%%%%%%%%%%%%SVM%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = [x((1:5000),:) ; x((5001:10000),:)];
label_vector=[ones(5000,1);-1*ones(5000,1)];
correct_test = ones(15000,1);m = 1;
for i = 1:2500
    i = m;
    correct_test(i,1) = 3;correct_test(i+1,1) = 1;correct_test(i+2,1) = 2;correct_test(i+3,1) = 3;correct_test(i+4,1) = 2;correct_test(i+5,1) = 1;
    m = i+6;
end
xi = 1;
for i = 1:15000
    if correct_test(i,1) == 1 
        new_data(xi,:) = y(i,:);
        xi = xi+1;
    elseif correct_test(i,1) == 2
        new_data(xi,:) = y(i,:);
        xi = xi+1;
    end
end
y = new_data;
a = [1;-1;-1;1];label_vector1 = repmat(a,2500,1);
min_traindata = min(x);
range = max(x) - min_traindata;
test_data = (y - repmat(min_traindata, size(y, 1), 1)) ./ repmat(range, size(y, 1), 1);
train_data = (x - repmat(min_traindata, size(x, 1), 1)) ./ repmat(range, size(x, 1), 1);
%linear
disp(' Linear Support Vector Machine');
training = svmtrain(label_vector,train_data, '-t 0 -b 0 -h 0');
[predicted_value]=svmpredict(label_vector1,test_data,training);
fcm = zeros(2,2);
for j = 1:10000
    firstindex = label_vector1(j,1) ;  secondindex = predicted_value(j,1) ;
    if firstindex == -1
        firstindex = 2;
    end
    if secondindex == -1
        secondindex = 2;
    end
    fcm(firstindex,secondindex) = fcm(firstindex,secondindex)+1;
end
w = training.SVs' * training.sv_coef;
b = -training.rho;
disp('Hyperplane Paramters');
disp('w');disp(w);disp('b');disp(b);
%RBF
disp(' RBF Support Vector Machine');
training1 = svmtrain(label_vector,train_data, '-t 2 -b 0 -h 0 -g 3 -c 6');
[predicted_value1]=svmpredict(label_vector1,test_data,training1);

fcm1 = zeros(2,2);
for j = 1:10000
    firstindex = label_vector1(j,1) ;  secondindex = predicted_value1(j,1) ;
    if firstindex == -1
        firstindex = 2;
    end
    if secondindex == -1
        secondindex = 2;
    end
    fcm1(firstindex,secondindex) = fcm1(firstindex,secondindex)+1;
end
w1 = training1.SVs' * training1.sv_coef;
b1 = -training1.rho;
disp('Hyperplane Paramters');
disp('w1');disp(w1);disp('b1');disp(b1);
timeelapsed = toc