import csv
a=[]
with open("enjoysport.csv","r")as csvfile:
    for row in csv.reader(csvfile):
        a.append(row)
    print(a)
print("\n Length of training instance are:",len(a))
num_att=len(a[0])-1
hypothesis=['0']*num_att
print("\n The initial hypothesis\n:",hypothesis)
for i in range(0,len(a)):
    if a[i][num_att]=="yes":
        for j in range(0,num_att):
            if hypothesis[j]=="0" or hypothesis[j]==a[i][j]:
                hypothesis[j]=a[i][j]
            else:
                hypothesis[j]="?"
    print("\n The hypothesis for the training instance {} is:\n".format(i+1),hypothesis)
print("\nThe maximallay specific hypothesis is\n",hypothesis)
