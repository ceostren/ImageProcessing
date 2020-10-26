import math
import random

def sqmul(x,expo,n):
    eh = bin(expo)
    eh = eh[2:]

    r = 1
    for i in range(0, len(eh)):
        r = r**2
        r = r % n
        if eh[i] == '1':
            r = r * x
            r = r % n
    return r

def isPrime(x):
    r = math.floor(math.sqrt(x))
    for i in range(2,r):
        if x % i == 0:
            return False
    return True

def FRT(p,s):
    t = False
    for i in range(0,s):
        a = random.randrange(2,p-2)
        c = math.gcd(a,p)
        if c == 1:
            if sqmul(a,p-1,p) != 1:
                return False
            else:
                t = True
    if not isPrime(p):
        return t
    return False


def carmTest(n):
    count1 = 5
    val1 = 10**n
    while count1 > 0:
        val1 -= 1
        t = FRT(val1,20)
        if t:
            count1 -= 1
    print(val1)



def q6():
    dict = {}
    for i in range(ord('A'),ord('Z')+1):
        key = sqmul(i,11,3763)
        val = chr(i)
        dict[key] = val
        print(val, " " , key)

    encoded = [2912,2929,3368,153,3222,3335,153,1222]
    str = ""
    for c in encoded:
        str += dict[c]
    print(str)

def almostIncreasingSequence(sequence[]):
    oneFail = false;
    for(i in i < sequence.length - 1 i++):
        if(sequence[i] >= sequence[i+1]):
            if(oneFail = true):
                return sequence[i];

            oneFail = true;
            sequence[i+1] = 4;


    return 0;






