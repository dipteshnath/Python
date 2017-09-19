
# coding: utf-8

# In[15]:


def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[int(len(arr) / 2)]# typecast to int from float
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


# In[19]:


print(quicksort([3,6,8,10,1,2]))


# In[29]:


x=3
print(type(x))
#print(type(float(x))
print(x+1)
print(x *2)
print(x**2)
x+=1
print(x)
y=2.5
print (type(y))
print(y,y+1,y**2)


# In[33]:


t=True
f=False
print(type(t))
print(t and f)
print(t or f)
print(not t)
print(t !=f)


# In[ ]:





# In[38]:


hello='Hello'
world='World'
print(hello)
print(len(hello))
hw=hello+' '+world
print(hw)
hw12='%s %s -----%d'% (hello,world,12)
print(hw12)


# In[45]:


s='hello'
print(s.capitalize())
print(s.upper())
print(s.rjust(8))
print(s.center(7))
print(s.replace('1','(ell)'))
print('   wor  ld'.strip())
print(s.count('l',3,4))
print(s.endswith('o'))


# In[119]:


print('01\t012\t0123\t01234'.expandtabs())
print(s.find('ll'))
print(s.find('ll',3))
print("The sum of 1 + 3 is {0}".format(9+2))
print(s.index('ll'))
print(s.isalnum())
print(s.isalpha())
print(s.isdigit())
print(s.islower())
print(s.istitle())
print(s.isupper())
#print(s.ljust())
print(s.lower())
print(s.lstrip())
print('www.example.com'.lstrip('cmowz.'))
print(s.rfind('l',1,5))
print(s.rindex('l',1,5))# like rfind by=ut raises value error
print(s.rjust(10,'#'))
print(s.rpartition('ll')[0])
print(s.rpartition('ll'))
print('hello lula lolo'.rsplit('l'))
print('   spacious   '.rstrip())
print('mississippi'.rstrip('ipz'))
print('1<>2<> <>3'.split())
print('1<>2<> <>3'.split('<>'))
print('1<>2<> <>3'.split('<>',1))
print('ab c\n ffr\r\n'.splitlines())#python recognizes \n and \r as line boundary
print(s.startswith('H'))
print('   we   '.strip())
print('www.example.com'.strip('cmowz.'))
print(s.swapcase())
print(s.title())
print(s.title().swapcase())
#print('read this shor text'.translate('aeiou'))
print(s.title().swapcase().upper())
print('1019'.isdecimal)
print('1019'.isnumeric())
print('1019'.isdecimal())


# In[134]:


xs=[3,1,2,2,2]
print(xs,xs[2])
print(xs[-1])
xs[2]='foo'
print(xs)
xs.append('bar')
print(xs)
print(xs.insert(len(xs),'sar'))#same as append
print(xs.index(3))
print(xs.count(2))
x=xs.pop()
print(x,xs)
a=[22,3.4,-10,3]
a.sort()
print(a)
a.reverse()
print(a)


# In[145]:


num=range(5)
print(num)
print(num[2:4])
print(num[2:])
print(num[:])
print(num[:-1])
#num[2,3]=[8,9]
print(num)


# In[154]:


animals=['cat','dog','monkey']
for animal in animals:
    print(animal)
    
for idx,animal in enumerate(animals):
    print('#%d: %s' % (idx+1,animal))

nums=[0,2,4,6,3]
squares=[]
for x in nums:
    squares.append(x**2)
print(squares)
sq1=[x**2 for x in nums]
print(sq1)
sq2=[x for x in nums if x%2==0]
print(sq2)


# In[159]:


d={"cat":"cute","dog":"bitch","god":"abstract"}
print(d["cat"])
print('cat' in d)
d['fish']='wet'
print(d)
print(d.get('monkey','N/A'))
print(d.get('fish','N/A'))


# In[173]:


d={'person':2,'cat':4,'fish':0,'spider':8}
for animal in d:
    legs=d[animal]
    print('A %s has %d legs' % (animal,legs))
for animal,legs in d.items():
    print('A %s has %d legs'%(animal,legs))

nums=[0,1,2,3,4]
even_num_to_square={x:x**2 for x in num if x%2==0}
print(even_num_to_square)
animals={'cat','dog'}
print('cat' in animals)
animals.add('fish')
print('fish' in animals)
print(len(animals))
for idx,animal in enumerate(animal):
    print('serial number %d : %s' % (idx+1,animal))


# In[174]:


#from math import sqrt
#nums={int(sqrt(x)) for x in range(30)}
#print nums


# In[180]:


# tuples similar to list. tuples can be used as keys in dictionary
d={(x,x+1):x for x in range(10)}
t=(5,6)
print(type(t))
print(d[t])
print(d[(1,2)])


# In[183]:


def sign(x):
    if x>0:
        return 'positive'
    elif x<0:
        return 'negative'
    else:
        return 'zero'
    
for x in[-1,0,1]:
    print(sign(x))
    
def hello(x, default=False):
    if default:
        print('HELLO %s'% x.upper())
    else:
        print('Hello %s '% x)

hello('Bob')
hello('Bob',default=True)


# In[185]:


class Greeter(object):
    def _init_(self,name):
        self.name=name
    def greet(self,loud=False):
        if loud:
            print('HELLO %s'%self.name.upper())
        else:
            print('Hello %s'% self.name)
g=Greeter('Fred')
g.greet()
g.greet(loud=True)


# In[191]:


import numpy as np
a=np.array([1,2,3])
print(type(a))
print(a.shape)
print(a[0],a[1],a[2])
a[0]=5
print(a)

b=np.array([[1,2,3,],[4,5,6]])
print(b.shape)
print(b[0,0],b[0,1])


# In[200]:


a=np.zeros((2,2))
print(a)
b=np.ones((1,2))
print(b)
c=np.full((2,2),5)
print(c)
d=np.eye(2)
print(d)
e=np.random.random((2,2))
print(e)


# In[202]:


import numpy as np
a=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
b=a[:2,1:3]
print(b)


# In[18]:


def isUnique(str):#Check if a string has unique character
    list1=[False]*128
    if(len(str)>128):
        return False
    for x in str:
        val=ord(x)
        if(list1[val]):
            return False
        list1[val]=True
    return True

    


# In[19]:


isUnique('abcaa')


# In[20]:


def isUnique(str):
    checker=0
    for x in str:
        val=ord(x)-ord('a')
        if(checker&(val<<1)>0):
            return False
        checker=checker|(val<<1)
    return True


# In[22]:


isUnique('abx')


# In[29]:


def sort(s):#Check if one string is permutation of other
    return ''.join(sorted(s))
def permutation(s,t):
    if(len(s)!=len(t)):
        return False
    return (sort(s) == sort(t))# return (sort(s) is sort(t)) gives always false


# In[33]:


permutation('abc','cbb')


# In[36]:


def permutation(s,t):
    letters=[0]*128
    if(len(s)!=len(t)):
        return False
    for c in s:
        val=ord(c)
        letters[val]=letters[val]+1
    for c in t:
        val=ord(c)
        letters[val]=letters[val]-1
        if letters[val]<0:
            return False
    return True


# In[38]:


permutation('abc','cba')


# In[5]:


def replacespace(str,trueLength):
    spaceCount=index=i=0
    for i in range(0,trueLength-1):
        if(str[i]==' '):
            spaceCount=spaceCount+1
    index=trueLength+spaceCount*2
    str+=' '*index
    if(trueLength<len(str)):str[trueLength]=''
    for i in reversed(range(0,trueLength-1)):
            if(str[i]==' '):
                str[index-1]='0'
                str[index-2]='2'
                str[index-3]='%'
                index=index-3
            else:
                str[index-1]=str[i]
                index=index-1
                
        


# In[6]:


replacespace('ab c',10)


# In[18]:


#Palindrome Permutation: Given a string, write a function to check if it is a permutation of
#a palindrome. A palindrome is a word or phrase that is the same forwards and backwards. A
#permutation is a rearrangement of letters. The palindrome does not need to be limited to just
#dictionary words.
def buildFrequencyTable(phrase):#Palindrome permutation
    table=[0]*(ord('z')-ord('a')+1)#table=[0]*26
    for c in phrase:
        x=(ord(c)-ord('a')) if (ord(c)>=ord('a') and ord(c)<=ord('z')) else -1
        if(x!=-1):
            table[x]+=1
    return table
def checkMaxOneOdd(table):
    x=[x for x in table if x%2==1]
    if(len(x)>1):return False
    else:return True
def isPermutationofPalindrome(s):
    table=buildFrequencyTable(s)
    return checkMaxOneOdd(table)


# In[20]:


isPermutationofPalindrome('txtaax')


# In[ ]:





# In[33]:


def checkPermutation(s):#String permutation
    chk=0
    for i in range(0,len(s)):
        chk=chk+abs(ord(s[i])-ord(s[-(i+1)]))
    if(chk==0):
        return True
    else:
        return False


# In[35]:


checkPermutation('abas')


# In[32]:


a='abcd'
print(a[0]-a[-1])


# In[21]:


grades={"Joel":80,"Tim":70}
a="Joel" in grades
print(a)
a=grades.get("ken",90)+8
print(a)
print(grades["Joel"]+1)
a=[1,2,3]


# In[25]:


from collections import Counter
c=Counter([0,1,2,0])

for word,count in c.most_common(2):
    print(word,count)
a={}


# In[28]:


pair=[x for x in range(10)]
print(pair)
def lazy_range(n):
 i=0
 while ii < n:
 yield(i)
 i += 1
 print(i)


# In[30]:


a=random.seed(10)


# In[32]:


import re
print(all([ # all of these are true, because
 not re.match("a", "cat"), # * 'cat' doesn't start with 'a'
 re.search("a", "cat"), # * 'cat' has an 'a' in it
 not re.search("c", "dog"), # * 'dog' doesn't have a 'c' in it
 3 == len(re.split("[ab]", "carbs")), # * split on a or b to ['c','r','s']
 "R-D-" == re.sub("[0-9]", "-", "R2D2") # * replace digits with dashes
 ])) # prints True


# In[12]:


#Power set code
def subs(l):
    if l == []:
        return [[]]

    x = subs(l[1:])
    print('>>',x)

    return x + [[l[0]] + y for y in x]
print (subs([1, 2, 3]))


# In[17]:


a=[1,2,3]
a[1:]
a[:0]


# In[18]:


def perms(s):        
    if(len(s)==1): return [s]
    result=[]
    for i,v in enumerate(s):
        print('x1',i)
        print('x2',v)
        result += [v+p for p in perms(s[:i]+s[i+1:])]
        print('x3',result)
    return result


perms('aac')


# In[23]:


fib={1:1,2:1}
print(fib[2])


# In[3]:


T=[0]*(2)
T


# In[7]:


def f2(n):
    T=[0]*(n+1)
    T[0]=T[1]=2
    for i in range(2,n+1):
        T[i]=0
        for j in range(1,i):
            T[i]+=2*T[j]*T[j-1]
    return(T[n])

print(f2(4))


# In[8]:


def f(n):
    T=[0]*(n+1)
    T[0]=T[1]=2
    T[2]=2*T[0]*T[1]
    for i in range(3,n+1):
        T[i]=T[i-1]+2*T[i-1]*T[i-2]
    return(T[n])
print(f(4))


# In[10]:


def isArrayInSortedOrder(A):
    if len(A)==1:
        return Ture
    return A[0]<=A[1] and isSorted(A[1:])
A=[123,4,2,4,22]
print(isArrayInSortedOrder(A))
print isSorted(A[1:])


# In[14]:


import re
str="an example word:cat!!"
match=re.search(r'word:\w\w\w',str)
if match:
    print('found',match.group())##'found word:cat'
else:
    print('did not find')


# In[28]:


class Node:
    # constructor
    def __init__(self, data):
        self.data = data
        self.next = None
         # method for setting the data field of the node    
    def set_data(self, data):
        self.data = data
    # method for getting the data field of the node   
    def get_data(self):
        return self.data
      # method for setting the next field of the node
    def set_next(self, next):
        self.next = next
       # method for getting the next field of the node    
    def get_next(self):
        return self.next
    # returns true if the node points to another node
    def has_next(self):
            return self.next != None


# In[29]:


class LinkedList(object):
     
    # initializing a list
    def __init__(self):
        self.length = 0
        self.head = None
         
    # method to add a node in the linked list
    def addNode(self, node):
        if self.length == 0:
            self.addBeg(node)
        else:
            self.addLast(node)
    def addBeg(self, node):
        newNode = node
        newNode.next = self.head
        self.head = newNode
        self.length += 1


# In[30]:


node1 = Node(1)
node2 = Node(2)
node3 = Node(3)
node4 = Node(4)
node5 = Node(5)
ll = LinkedList()
ll.addNode(node1)
ll.addNode(node2)
ll.addNode(node3)
ll.addNode(node4)
ll.addNode(node5)


# In[1]:


get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
import numpy as np


# In[13]:


from sklearn.preprocessing import PolynomialFeatures
x=np.array([2,3,4])
poly=PolynomialFeatures(3,include_bias=False)
poly.fit_transform(x[:,None])


# In[16]:


rng=np.random.RandomState(1)
x=10*rng.rand(50)
y=2*x-5+rng.randn(50)
plt.scatter(x,y);


# In[22]:


from sklearn.linear_model import LinearRegression
model=LinearRegression(fit_intercept=True)
model.fit(x[:,np.newaxis],y)
xfit=np.linspace(0,10,1000)
yfit=model.predict(xfit[:,np.newaxis])

plt.scatter(x,y)
plt.plot(xfit,yfit)


# In[14]:


from sklearn.pipeline import make_pipeline
poly_model=make_pipeline(PolynomialFeatures(7),LinearRegression())


# In[ ]:





# In[23]:


print("Model slope: ",model.coef_[0])
print("Model intercept: ",model.intercept_)


# In[ ]:





# In[24]:


rng=np.random.RandomState(1)
x=10*rng.rand(50)
y=np.sin(x)+0.1*rng.randn(50)
poly_model.fit(x[:,np.newaxis],y)
yfit=poly_model.predict(xfit[:,np.newaxis])
plt.scatter(x,y)
plt.plot(xfit,yfit)


# In[1]:


import numpy as np
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)


# In[6]:


y[1]


# In[50]:


def reverse(text):
    if len(text) <= 1:
        return text

    return reverse(text[1:]) + text[0]

def  luggage(weights):
    ans = ""
    b=reverse(weights)
    a=b.split(',')
    for i in range(0,len(a)-2):
        if(i%3==0):
         a[i],a[i+2]=a[i+2],a[i]
        ans = ','.join(a)
    return(ans)


# In[51]:


luggage("1,2,3,4,5,6,7,8,9")


# In[1]:


print("asd")


# In[4]:



def  consecutive(num):
 cnt=0
for i in range(1,num):
    x=0
    for j in range(i+1,num):
        x=x+i+j
        if(x==num):
            cnt++
return cnt


# In[5]:


a=123
a.split()


# In[7]:


len(str(123))

