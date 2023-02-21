T=int(input())
for i in range(T):
    n=int(input())
    A=list(map(int,input().split()))
    ind=[]
    summ=0
    for k in range(n):
        if A[k]==0:
            ind.append(k)
        summ+=A[k]
    if summ+len(ind)==0:
        print(len(ind)+1)
    else:
        print(len(ind))