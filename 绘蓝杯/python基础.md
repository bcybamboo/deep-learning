[TOC]



#### python输入输出

##### 输出  print()函数

`print()`函数也可以接受多个字符串，用逗号“,”隔开，就可以连成一串输出

`print()`会依次打印每个字符串，遇到逗号“,”会输出一个空格

`print()`也可以打印整数，或者计算结果

##### 输入  input（）函数

先存在一个变量里

```python
name=input()
```

`input()`函数的返回类型是`str`,不能直接和整数比较

##### 输出方式有两种：

​	1.直接输入变量名  >>>name

​	2.用输出函数 >>>print(name)

#### 数据类型和变量

整数运算永远是精确的（除法也是精确的），而浮点数运算则可能会有四舍五入的误差

字符串  用‘或“括起来

1.如果单引号本身也是一个字符，那么用双引号括起来

2.如果单、双引号都有，用转义字符\来标识

![image-20221020173141729](C:\Users\bamboo\AppData\Roaming\Typora\typora-user-images\image-20221020173141729.png)

3.转义字符`\`可以转义很多字符，比如`\n`表示换行，`\t`表示制表符，字符`\`本身也要转义，所以`\\`表示的字符就是`\`

如果前面加上r，默认不转义

```python
>>> print('\\\t\\')
\       \
>>> print(r'\\\t\\')
\\\t\\
```

*print(r'\t\')显示错误print(r'\t\\')就代码通过*

原因：因为第一个公式最后一个 (’) 这个前面用了转义符（\）,就改变了（’) 本身的符号意义，不会被当做（ r' ' )这整个符号来处理了，会把后面的单引号当做一个文本字符，相当于（r' )只有一个单引号，符号不完整了。

4.'''...'''格式表示多行内容



空值   None并不是0，是一个特殊的空值



布尔值  用and or not计算（常用在条件判断）

`and`  同为True结果才是True

`or`  同为False结果才是False

`not` 是单目运算符，把`True`变成`False`，`False`变成`True`

除法

1.`\`		计算结果是浮点数

2.`\\\` 	计算结果是整数，只取结果的整数部分

余数运算%  

对变量赋值`x = y`是把变量`x`指向真正的对象，该对象是变量`y`所指向的。随后对变量`y`的赋值*不影响*变量`x`的指向

#### 字符串和编码

ord()函数获取字符的整数表示

chr()函数把编码转换为对应的字符

![image-20221021154640510](C:\Users\bamboo\AppData\Roaming\Typora\typora-user-images\image-20221021154640510.png)

把`bytes`变为`str`

```python
>>> b'\xe4\xb8\xad\xe6\x96\x87'.decode('utf-8')
'中文'
```

`str`变成`bytes`

```
>>> '中文'.encode('utf-8')
b'\xe4\xb8\xad\xe6\x96\x87'
```

![image-20221021212249936](C:\Users\bamboo\AppData\Roaming\Typora\typora-user-images\image-20221021212249936.png)

若bytes包含无法解码的字节，会报错

如果只有一小部分无效字节，可以传入`errors='ignore'`忽略错误的字节

```python
>>> b'\xe4\xb8\xad\xff'.decode('utf-8', errors='ignore')
'中'
```

`len()`函数   计算str包含多少个字符

`len()`函数计算的是`str`的字符数，如果换成`bytes`，`len()`函数就计算字节数

一个中文字符占3个字节，一个英文字符占1个字节

格式化

方法一：符号%

```python
>>>'Hello %s,you have $%d'%('name',10)
'Hello name,you have $10'
```

%s字符串替换，%d整数替换

一一对应，变量只有一个，可以省略（）

常见占位符

| 占位符 |   替换内容   |
| :----: | :----------: |
|   %d   |     整数     |
|   %f   |    浮点数    |
|   %s   |    字符串    |
|   %x   | 十六进制整数 |

![5918cccaa62b0c3b904fff283c384a8e.png](https://img-blog.csdnimg.cn/img_convert/5918cccaa62b0c3b904fff283c384a8e.png)

```python
PI=3.1415926
print("pi1 = %10.3f" % PI)
# 总宽度为10，小数位精度为3
print("pi2 = %.*f" % (3, PI))
# *表示从后面的元组中读取3，定义精度
print("pi3 = %010.3f" % PI)
# 用0填充空白
print("pi4 = %-10.3f" % PI)
# 左对齐，总宽度10个字符，小数位精度为3
print("pi5 = %+f" % PI)
# 在浮点数前面显示正号
```

结果

pi1 =      3.142
pi2 = 3.142
pi3 = 000003.142
pi4 = 3.142     
pi5 = +3.141593

![42201accd916692ef79569d53a4c3b65.png](https://img-blog.csdnimg.cn/img_convert/42201accd916692ef79569d53a4c3b65.png)



方法二：format()字符串函数

1.（通过位置索引值）

```python
print('{0}{1}'.format('python',3.9))
print('{}{}'.format('python',3.9))
print('{0}{1}{0}'.format('python',3.9))

```

结果

python3.9
python3.9
python3.9python

说明位置自定义后可以重复引用

2.（通过关键字索引值）

```
print('{name}年龄是{age}岁'.format(age=18,name="Q"))
```

Q年龄是18岁

3.（通过下标进行索引）

```
L=["Jason",30]
print('{0[0]}年龄是{0[1]}岁。'.format(L))
```

Jason年龄是30岁。

![b71e054a865e8414bb8495fe3814ff95.png](https://img-blog.csdnimg.cn/img_convert/b71e054a865e8414bb8495fe3814ff95.png)

通常用0、*、#、@进行填充，默认为空格

方法三：`f-string`字符串

```
>>> print(f'The area of a circle with radius {r} is {s:.2f}')
The area of a circle with radius 2.5 is 19.62
```

#### list和tuple

##### list（中括号）

和数组很像

```
>>>a=['q','w','e']
#访问每个位置的元素
>>>a[0]
'q'
#计算个数
>>>len(a)
3

```

不能越界

可以倒推，但也不能越界

```
>>>a[-1]
'e'
```

追加元素`append('元素')`

```
>>>a.append('r')
>>>a
['q','w','e','r']
```

插入元素到指定位置`insert(i,'元素')`

```
>>>a.insert(1,'t')#1表示索引号为1，从0开始
>>>a
['q', 't', 'w', 'e']
```

删除末尾元素`pop()`

```
>>>a.pop()
'e'
>>> a
['q', 't', 'w']
```

删除指定位置的元素`pop(i)`

```
>>>a.pop(1)
't'
>>> a
['q', 'w']
```

替换元素

```
>>>a[1]='u'
>>>a
['q','u']
```

里面元素数据类型可以不同

list可以嵌套，此时可以看成二维数组或多维数组

##### tuple（小括号）

和list类似，但tuple一旦初始化就不能更改

```
>>> classmates = ('Michael', 'Bob', 'Tracy')
```

特殊：只有一个元素的tuple定义时要加一个逗号来消除歧义（显示时也会加逗号）

如果tuple中有list，它看起来就是内容可变的

#### 条件判断

```
if <条件判断1>:
    <执行1>
elif <条件判断2>:
    <执行2>
elif <条件判断3>:
    <执行3>
else:
    <执行4>
```

elif是else if的缩写

输入的input函数返回类型是str

所以要转换类型

```
s=input()
birth=int(s)
if birth<2000:
	print('00后')
```

#### 循环

1.`for x in...`循环

把每个元素代入变量x，然后执行缩进语句

#`range（）`生成整数序列，再通过list()函数转换成list

比如range(5)就是从0开始小于5的整数

```
>>>list(range(5))
[0,1,2,3,4]
```

2.`while`循环

`break`提前退出循环

`continue`跳过当前循环

#### 使用dict和set

##### `dict`

字典  速度快(用空间换时间)

把数据放入dict方法：1.初始化指定  2.通过key放入

一个key对应一个值，多次对一个key放入value,后面的值会把前面的覆盖掉

如果key不存在，dict就会报错

避免key不存在的错误

1.通过in判断key是否存在

```
>>> 'Thomas' in d
False
```

2.`get()`，不存在就返回`None`

```
>>> d.get('Thomas')
>>> d.get('Thomas', -1)
-1
```

删除key方法`pop(key)`

##### set

set中没有重复的key,会被过滤

```
>>>s=set([1,2,3])#需提供一个list作为输入集合
>>>s
{1, 2, 3}
>>>s=set([1,2,3,2,1,2,3])#重复会被过滤
>>>s
{1, 2, 3}
>>>s.add(4)#添加
>>>s
{1, 2, 3, 4}
>>>s.remove(4)#删除
>>>s
{1, 2, 3}
```

set可做交集、并集等操作

```
>>> s1=set([1,2,3])
>>> s2=set([2,3,4])
>>> s1&s2
{2, 3}
>>> s1|s2
{1, 2, 3, 4}
```

#### 函数

##### 调用函数

`abs`函数

```
>>>abs(-20)
20
```

`max`函数

可以接收任意多个参数并返回最大的那个

```
>>>max(2,3,1,-5)
3
```

`hex()`函数

十进制转成十六进制,以字符串形式表示

```
>>>hex(9)
'0x9'
```

##### 数据类型转换

```
>>> int(12.34)
12
>>> float('12.34')
12.34
>>> str(1.23)
'1.23'
>>> str(100)
'100'
>>> bool(1)
True
```

```
>>> a = abs # 变量a指向abs函数
>>> a(-1) # 所以也可以通过a调用abs函数
1
```

##### 定义函数

```
def 函数名（参数）：

Tab函数体

	返回值return
```

空函数

```
def nop():
	pass
```

`pass`语句什么都不做，占位

参数检查

个数不对，`TypeError`

参数类型不对就无法检查

用内置函数`isinstance()`实现只允许某种类型的参数

```
if not isinstance(x,(int,float)):
	raise TypeError('bad operand type')
```

返回多个值

函数可以同时返回多个值，但其实就是一个tuple

##### 函数参数

###### 位置参数  按位置顺序依次赋给参数

###### 默认参数  降低调用函数的难度

注意：1，必选参数在前，默认参数在后

​			2，默认参数必须指向不变对象

当传入的参数个数可变时，

1.可以用list或tuple传进去

```
def calc(number):
    sum=0
    for n in number:
        sum=sum+n*n

    return sum
 
a=calc([1,2,3])
print(a)
```

结果：14

###### 2.或者用可变参数，仅仅在参数前面加一个*号

```
def calc(*number):
    sum=0
    for n in number:
        sum=sum+n*n

    return sum
 
a=calc(1,2,3)
print(a)
```

结果：14

3.已经有一个list或tuple，在list或tuple前面加一个`*`号，把list或tuple的元素变成可变参数传进去（常见）

```
>>>nums=[1,2,3]
>>>calc(*nums)
14
```

###### 关键字参数

可变参数允许你传入0个或任意个参数，这些可变参数在函数调用时自动组装为一个tuple。而关键字参数允许你传入0个或任意个含参数名的参数，这些关键字参数在函数内部自动组装为一个dict。

```
def person(name,age,**other):
    print('name:',name)
    print('age:', age)
    print(other)

person('mm',24,city='BeiJing',job='teather')
```

结果：name: mm
			age: 24
			{'city': 'BeiJing', 'job': 'teather'}

name和age必填项，**other是选填项

也可以先装一个dict再转化成关键字参数

命名关键字参数

和关键字参数`**kw`不同，命名关键字参数需要一个特殊分隔符`*`，`*`后面的参数被视为命名关键字参数。

```
def person(name, age, *, city, job):
    print(name, age, city, job)
```

如果函数定义中已经有了一个可变参数，后面跟着的命名关键字参数就不再需要一个特殊分隔符`*`了

命名关键字参数必须传入参数名，这和位置参数不同。如果没有传入参数名，调用将报错

**参数定义的顺序必须是：必选参数、默认参数、可变参数、命名关键字参数和关键字参数**

`*args`是可变参数，args接收的是一个tuple；

`**kw`是关键字参数，kw接收的是一个dict。

##### 递归函数

函数在内部调用自身，就是递归函数（定义简单，逻辑清晰）

```
def fact(n):
    if n==1:
        return 1
    return n*fact(n-1)
```

*使用递归函数要注意防止栈溢出！在计算机中，函数调用是通过栈（stack）这种数据结构实现的，每当进入一个函数调用，栈就会加一层栈帧，每当函数返回，栈就会减一层栈帧。由于栈的大小不是无限的，所以，递归调用的次数过多，会导致栈溢出。

解决递归调用栈溢出的方法：尾递归优化（循环也算是一种特殊的尾递归函数）

 尾递归是指，在函数返回的时候，调用自身本身，并且，return语句不能包含表达式。

python解释器没有做优化，还是会溢出

#### 高级特性

##### 切片`Slice`

方便取一个list或tuple的部分元素

```
L=list(range(100))
print(L[0:10])
#L[0:10]表示从索引0开始到索引10（但不包括索引10），即索引0~9十个元素
#如果第一个索引为0，可以省略，第一个索引也可以为其他数字

```

同时也支持倒数切片

tuple、字符串也可以切片操作，但操作结果还是它本身

###### [:-1]与[::-1]区别

**b = a[i:j]**

**表示复制a[i]到a[j-1]，以生成新的list对象**



**b = a[i:j:s]**

**表示：i,j与上面的一样，但s表示步进，缺省为1.
所以 a[i:j:1] 相当于 a[i:j]**

**当s<0时，i缺省时，默认为-1. j缺省时，默认为-len(a)-1**


**所以a[::-1]相当于 a[-1:-len(a)-1:-1]，也就是从最后一个元素到第一个元素复制一遍，即倒序**

迭代

如果给定一个`list`或`tuple`，我们可以通过`for`循环来遍历这个`list`或`tuple`，这种遍历我们称为迭代（Iteration）

只要作用与一个可迭代对象，for循环就可以运行

**判断可迭代对象方法：`collections.abc`模块的`Iterable`类型判断**

```
>>>from collections.abc import Iterable
>>>isinstance('abc',Iterable)#判断str是否可迭代
Ture
```

实现下标循环，用python内置的`enumberate`函数可以把list 变成索引-元素对

```
for i,value in enumerate(['A','B','C']):
    print(i,value)
0 A
1 B
2 C
```

引用2个变量也是很常见的

##### 列表生成式

把要生成的元素`x * x`放到前面，后面跟`for`循环

在一个列表生成式中，`for`前面的`if ... else`是表达式得出一个结果，而`for`后面的`if`是过滤条件，不能带`else`

```
>>> [x * x for x in range(1, 11)]
[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
>>> [x * x for x in range(1, 11) if x % 2 == 0]
[4, 16, 36, 64, 100]
```

还可以使用两层循环生成全排列

```
>>>[m+n for m in 'ABC' for n in 'XYZ']
['AX', 'AY', 'AZ', 'BX', 'BY', 'BZ', 'CX', 'CY', 'CZ']
```

把list所有字符串变成小写

```
>>> L = ['Hello', 'World', 'IBM', 'Apple']
>>> [s.lower() for s in L]#s就是字符串
['hello', 'world', 'ibm', 'apple']
```

扩展

```
print(str.upper())          # 把所有字符中的小写字母转换成大写字母
print(str.lower())          # 把所有字符中的大写字母转换成小写字母
print(str.capitalize())     # 把第一个字母转化为大写字母，其余小写
print(str.title())          # 把每个单词的第一个字母转化为大写，其余小写 
```

##### 生成器generator

一边循环一边计算，节省空间

方法一：把一个列表生成式的`[]`改成`()`

打印generator元素     

1.next（)获得下一个返回值，没有更多的元素时，*抛出`StopIteration`的错误*。（基本不用）

2.for循环

for循环实质就是通过不断调用next()函数实现的

函数定义中包含yield就是generator函数，输出一个返回一个

每次调用`next()`的时候执行，遇到`yield`语句返回，再次执行时从上次返回的`yield`语句处继续执行

**注意：调用generator函数会创建一个generator对象，多次调用generator函数会创建多个相互独立的generator。**

```
#False
>>>next(函数名())
第一个元素
>>>next(函数名())
第一个元素
#创建了多个相互独立的generator
#Ture：创建一个generator对象，然后不断对这一个generator对象调用next()
>>>g=函数名()
>>>next(g)
第一个元素
>>>next(g)
第二个元素
```

用`for`循环调用generator时，发现拿不到generator的`return`语句的返回值。如果想要拿到返回值，必须捕获`StopIteration`错误，返回值包含在`StopIteration`的`value`中

用generator来实现杨辉三角

```python
def triangles():
    L = [1]
    while True:
        yield L
        L1=[L[i]+L[i + 1] for i in range(len(L) - 1)]#前两项相加（列表生成式）
        L = [1] + L1 + [1]
        
n = 0#用来终止循环
results = []#定义一个结果list
for t in triangles():
    results.append(t)#加一个结果
    n = n + 1
    if n == 10:
        break

for t in results:
    print(t)    #generator需要用for循环来输出    
```

要理解generator的工作原理，它是在**`for`循环**的过程中不断计算出下一个元素，并在适当的条件结束`for`循环。对于**函数改成的generator**来说，遇到`return`语句或者执行到函数体最后一行语句，就是结束generator的指令，`for`循环随之结束。

##### 迭代器

生成器都是`Iterator`对象，表示惰性计算序列，但`list`、`dict`、`str`虽然是`Iterable`，却不是`Iterator`。

把`list`、`dict`、`str`等`Iterable`变成`Iterator`可以使用`iter()`函数。

```
def ange_for(i):
    it = iter(i)  # iter(i) 根据i获得一个Iterator对象
    while True:
        try:
            x = next(it)
        except StopIteration:
            break
```

检查对象x能否迭代，最准确的方法是：调用iter(x)函数，如果不可迭代，则会抛出异常：“TypeError: 'C' is not iterable.”。

##### 高阶函数

变量可以指向函数，函数的参数能接收变量，那么一个函数就可以接收另一个函数作为参数，这种函数就称之为高阶函数(函数里面有函数)

```
def add(x, y, f):
    return f(x) + f(y)
print(add(-5, 6, abs))
结果：11
```

###### map(函数，序列)函数

`map()`函数接收两个参数，一个是函数，一个是`Iterable`，`map`将传入的函数依次作用到序列的每个元素，并把结果作为新的`Iterator`返回

```
def f(x):
	return x*x
r=map(f,[1,2,3,4,5,6,7,8,9])
list(r)
[1, 4, 9, 16, 25, 36, 49, 64, 81]
```

把list中每个元素用f函数过一遍后得到一个新值，然后再重新组成一个新list

```
def add(x):
    return x**2			#计算x的平方

lists = range(11)       #创建包含 0-10 的列表
a = map(add,lists)      #计算 0-10 的平方，并映射
print(a)                # 返回一个迭代器：<map object at 0x0000025574F68F70>
print(list(a))          # 使用 list() 转换为列表。结果为：[0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

#使用lambda匿名函数的形式复现上面的代码会更简洁一些

print(list(map(lambda x:x**2,range(11))))   # 结果为：[0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
```

**map()函数不改变原有的 list，而是返回一个新的 list**

###### reduce累积

`reduce`把一个函数作用在一个序列`[x1, x2, x3, ...]`上，这个函数**必须接收两个参数**，`reduce`把结果继续和序列的下一个元素做累积计算

```
reduce(f, [x1, x2, x3, x4]) = f(f(f(x1, x2), x3), x4)
```

```
def prod(L):
    return reduce(lambda x,y:x*y,L)
print('3 * 5 * 7 * 9 =', prod([3, 5, 7, 9]))#累积
```

###### filter筛选函数

只接受一个函数和一个序列，根据返回值是`True`还是`False`决定保留还是丢弃该元素。

返回`Iterator`是一个惰性序列，需要list函数获得所有结果并返回list。

```
def is_odd(n):
    return n % 2 == 1（判断条件）

list(filter(is_odd, [1, 2, 4, 5, 6, 9, 10, 15]))
# 结果: [1, 5, 9, 15]
```

###### sorted排序函数（可以对list中的数字、字符串排序）

```
>>>sorted([23,43,12,-1,8])#对数字排序
[-1, 8, 12, 23, 43]
>>> sorted([-35,-89,1,0,45,16],key=abs)
[0, 1, 16, -35, 45, -89]
#通过key函数把list中元素都处理一遍，形成新的list名为key，然后sorted函数再对key函数排序
>>> sorted(['abd','Zoo','weer'],key=str.upper)
['abd', 'weer', 'Zoo']#不在乎大小写（全变小写或大写）
>>> sorted(['abd','Zoo','weer'],key=str.upper,reverse=True)
['Zoo', 'weer', 'abd']#加上reverse=True就倒序
```

sort和reverse区别：

sort排序结果不能逆转，reverse再次reverse就可以逆转

##### 返回函数

在函数里面定义函数，返回的是函数f，调用函数f时出现计算结果

每次调用都会创建一个新的函数，但并非立刻执行，而是等到所有函数返回后再计算

返回函数不能引用任何循环变量或者可变量

```
def count():
    def f(j):
        def g():
            return j*j
        return g
    fs=[]    #重新创建一个新list
    for i in range(1,4):
        fs.append(f(i))
    return fs
f1,f2,f3=count()
print(f1(),f2(),f3())
```

如果内层函数需要用外层变量，加上`nonlocal x`

##### lambda匿名函数

```
lambda x:x*x
lambda x,y:x*y
```

##### decorator装饰器(不改变原函数并同时处理多个函数是比较方便)

**@函数名**
def xxx():
	pass
#python内部会自动执行 函数名(xxx),执行完之后，再讲结果赋值给xxx.

相当于**xxx=函数名(xxx)**

```
#需求：开始输出begin,结束输出end
import functools#导入functools模块

def outer(origin):#origin=func
   def decorator(func):
   @functools.wraps(func)#消除装饰器对原函数造成的影响
   		def inner(*args,**kw):
        	print('begin')#执行前操作
        	res=origin(*args,**kw)#支持传参，调用原来的func函数
        	print('end')#执行后操作
        	return res
    	return inner#内层函数一定不加括号

@outer #相当于func=outer(func)
def func(a1):
    print('我是func函数')
    value=(11,22,33,44)
    return value
    
res=func(1)
print(res)
```

函数不加括号是调用，加了括号是执行

##### `functools.partial`偏函数（函数参数太多需要简化）

把一个函数的某些参数**设置默认值**，返回一个新函数，使调用新函数更简单

```
import functools
int2=functools.partial(int,base=2)#转化二进制
#base相当于关键字参数
```

#### 模块

创建模块名不能与python自带的模块名称冲突。检查方法是在Python交互环境执行`import abc`，若成功则说明系统存在此模块。

每一个包目录下面都会有一个`__init__.py`的文件，这个文件是必须存在的，否则，Python就把这个目录当成普通目录，而不是一个包。

##### 使用模块

```
#!/usr/bin/env python3        让这个hello.py文件直接在Unix/Linux/Mac上运行
# -*- coding: utf-8 -*-       表示.py文件本身使用标准UTF-8编码

' a test module '             表示模块的文档注释

__author__ = 'name'          作者名
```

if __name__ == '__main__',若为主函数，则执行下列main内容；若是import模块，跳过main内容

import x 是执行x模块中非main函数的部分，是执行公开函数

私有函数可以加前缀 __，表示不允许外部访问。

加前缀__表示私有函数，不允许外部访问。

外部不需要函数定义成私有，需要引用定义成公开

#### 面向对象编程

```
class Person(object):
#初始化你将创建实例的属性
	def __init__(self,a,b,c):
		self.a=a
		self.b=b
		self.c=c
#定义你将创建的实例所有用的技能
	def xxx(self):
		print('xxxxx')
#开始创建实例
ccc=Person(a,b,c)#具体数字

#你的实例开始使用它的技能
ccc.xxx()
```

##### 类和实例

类是抽象的模板，实例是根据类创建出来的一个个具体的“对象”，每个对象都有相同的方法，但数据可能不同。

class  类名（object）

类名通常为大写字母开头的单词

object表示从哪个类继承下来的，通常，如果没有合适的继承类，就使用`object`类，这是所有类最终都会继承的类。

**创建实例**是通过**类名＋（）**实现的

__init__方法第一个参数永远是**self**,表示创建的实例本身，并且调用时不用传递参数。

###### 数据封装

```
	def __init__(self,a,b,c):
		self.__a=a
		self.__b=b
		self.__c=c
```

属性名字前加上两个下划线，变成私有变量，无法从外界访问self.——a和self.——b

优点：方便检查，避免传入无效参数

获取方法

```
def get_a(self):
	return self.__a

def get_b(self):
	return self.__b
```

修改方法

```
def set_b():
	self._b=b
```

##### 继承和多态

对于`Person`来说，`object`就是他的父类。对于`object`来说，`Person`就是他的父类。子类获得了父类全部功能。

定义一个class相当于定义了一种数据类型。

 任何依赖`Animal`作为参数的函数或者方法都可以不加修改地正常运行，原因就在于多态。

多态只管调用，不管细节。

继承可以把父类的所有功能都直接拿过来，这样就不必重零做起，子类只需要新增自己特有的方法，也可以把父类不适合的方法覆盖重写。

动态和静态语言

静态必须要传入子类，动态能跑就行

```
class Animal(object):   #编写Animal类
    def run(self):
        print("Animal is running...")

class Dog(Animal):  #Dog类继承Amimal类，没有run方法
    pass

class Cat(Animal):  #Cat类继承Animal类，有自己的run方法
    def run(self):
        print('Cat is running...')
    pass

class Car(object):  #Car类不继承，有自己的run方法
    def run(self):
        print('Car is running...')

class Stone(object):  #Stone类不继承，也没有run方法
    pass

def run_twice(animal):
    animal.run()
    animal.run()

run_twice(Animal())
run_twice(Dog())
run_twice(Cat())
run_twice(Car())
run_twice(Stone())
```

除了stone不会跑，能跑的都是好“鸭子”

##### 获取对象信息

使用`type()`判断对象基本类型

使用`isinstance()`可以判断class的类型，并且能用`type()`判断的也能用`isinstance()`判断，还可以判断一个变量是否时某些类型的一种

使用`dir()`获得一个对象的所有属性和方法

测试该对象的属性

```
>>> class Myobject(object):
...     def __init__(self):
...             self.x=9
...     def power(self):
...             return self.x*self.x
...
>>> obj=Myobject()
>>> hasattr(obj,'x')#有没有x呢
True
>>> obj.x
9
>>> setattr(obj,'y',19)#设置一个属性y
>>> getattr(obj,'y')#获取属性y
19
>>> getattr(obj,'q',404)#获取属性q，如果不存在，返回默认值404
404
```

实例属性和类属性不能用相同的名字

实例属性是各个实例所有，互不干扰

类属性属于类所有，所有实例共享

（类是食谱，实例是做出来的美食）
