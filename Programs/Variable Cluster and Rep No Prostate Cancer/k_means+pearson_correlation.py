import numpy as np

from scipy.stats import pearsonr

from Bio.Cluster import kcluster
from Bio.Cluster import clustercentroids

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import naive_bayes as nb
from sklearn import svm

print "Stage 0: Initiating Data Dimention Reduction Using K-means + Pearson Correlation"
choice=input("\nChoose:\n1: KNN\n2: SVM\n3: Gaussian NBC\n")

f = open('prostate', 'r')
dim=map(int,f.readline().split())
k_fold_number=dim[0]
X_features=[]
Y_class=[]

for i in range(dim[0]):
    x=f.readline()
    x=list(x.split())
    #x.pop()
    Y_class.append(x.pop())
    X_features.append(map(float,x))

f.close()
Y_class=map(float,Y_class)
print "\nData Load Complete\nAccuracies:\n\n"

target=Y_class
cluster_number=1 # change the number of clusters as required
for i in [1,2,3,4,5,10,15,20,25,30]:
    cluster_rep=i
    dst =[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    corr=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    dataset=[]

    clusterid,error,nfound=kcluster(X_features,cluster_number,None,None,1,1,'a','c',None)

    X_features=np.transpose(X_features)

    for i in range(len(clusterid)):
        dst[clusterid[i]].append(X_features[i])


    X_features=np.transpose(X_features)
    cdata,cmask=clustercentroids(X_features,None,clusterid,'a',1)
    cdata=np.transpose(cdata)

    
    for i in range(len(corr)):
        for j in range(len(dst[i])):
            cr,p_val=pearsonr(dst[i][j],cdata[i])
            corr[i].append(cr)
    
    ddst=[]
    accuracy_final=0.0
    
    for i in range(cluster_rep):
        done=False
        for j in range(len(dst)):
            if i>len(dst):
                done=True
                break
            sorted_cdata=sorted(range(len(corr[j])),key=lambda x:corr[j][x],reverse=True)
            for k in range(len(sorted_cdata)):
                ddst.append(dst[j][sorted_cdata[k]])
                kf = KFold(n_splits = k_fold_number, shuffle = True)
                accuracies = []
                scores = []
                dataset=np.transpose(ddst)
                for it in range(k_fold_number):
                    for train, test in kf.split(dataset):
                        train_set = []
                        train_labels = []
                        test_set = []
                        test_labels = []
                        for i in train:
                            train_set.append(dataset[i])
                            train_labels.append(target[i])
                        for i in test:
                            test_set.append(dataset[i])
                            test_labels.append(target[i])
                            
                        if(choice==1):
                            classifier = KNeighborsClassifier()
                        elif(choice==2):
                            classifier = svm.SVC()
                        elif(choice==3):
                            classifier = nb.GaussianNB()
                        elif (choice==4):
                            classifier = nb.MultinomialNB()

                        predicted = classifier.fit(train_set, train_labels).predict(test_set)

                        incorrect = (test_labels != predicted).sum()
                        accuracy = ((len(test_set) - incorrect)*100) / len(test_set)
                        accuracies.append(accuracy)
    
                if (sum(accuracies)/len(accuracies)<=accuracy_final):
                    ddst.pop()
                else:
                    accuracy_final=sum(accuracies)/len(accuracies)
                    break
        if done:
            break
    print str(accuracy_final)+" %"
a=raw_input("\n\nPress any button to exit")
