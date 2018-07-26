import math
def isPrime(n):
    # 1 represents n is a prime,0 represents n is not a prime
    # I don't carry out input anomaly detection. n>0
    if n==2 or n==3:
        return 1
    if n%6 !=1 and n%6!=5 or n==1:
        return 0
    n_sqrt=int(math.sqrt(n))
    for i in range(5,n_sqrt,6):
        if n%i==0 or n%(i+2)==0:
            return 0
    return 1

oneZero=isPrime(1)
print(oneZero)