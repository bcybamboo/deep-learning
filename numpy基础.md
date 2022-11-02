[TOC]



# 数值类型

## dtype数据类型

![](C:\Users\bamboo\AppData\Roaming\Typora\typora-user-images\image-20221030161750596.png)

都属于dtype(data-type)对象的实例

用numpy.dtype(object,align,copy)来指定数据类型

在数组里用`dtype=`参数

## dtype数据类型转换.astype()

```
import numpy as np

a=np.array([1.1,2.2,3.3],dtype=np.float64)#定义float
print(a,a.dtype)

print(a.astype(int).dtype)#转成int类型
```

# ndarray数组类型

数组三种形式

列表：[1,2,3]

元组：(1,2,3,4,5)     元素不能修改

字典：{A:1,B:2}       由键和值构成

`ndarray`类的六个参数

`shape`形状         `dtype`数据类型         `buffer`对象暴露缓冲区接口        

`offset`数组数据偏移量             `strides`数据步长           

`order`{‘C','F'}，以行或列为主排列顺序

## 创建ndarray方法

### 方法一：使用python类型array

```
numpy.array(object,dtype=None,copy=True,order=None,subok=False,ndmin=0)
```

- `object`：列表、元组等。
- `dtype`：数据类型。如果未给出，则类型为被保存对象所需的最小类型。
- `copy`：布尔类型，默认 True，表示复制对象。
- `order`:创建数组的样式，C为行方向，F为列方向，A为任意方向（默认）。
- `r`：顺序。
- `subok`：布尔类型，表示子类是否被传递。
- `ndmin`：生成的数组应具有的最小维数。

```
np.array([[1,2,3],[4,5,6]])#列表创建
np.array([(1,2),(3,4),(5，6)])#列表和元组结合创建
```

### 方法二：arange方法创建（在给定区间内创建一系列均匀间隔的值）

```
numpy.arange(start, stop, step, dtype=None)
```

在[start,stop)中以step为步长新建数组

```
np.arange(1,9,2,dtype='float32')
```

### 方法三：linspace方法创建（在给定区间内返回间隔均匀的值）

```
numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
```

- `start`：序列的起始值。
- `stop`：序列的结束值。
- `num`：生成的样本数。默认值为 50。
- `endpoint`：布尔值，如果为真，则最后一个样本包含在序列内。
- `retstep`：布尔值，如果为真，返回间距。
- `dtype`：数组的类型。

```
#在0~10之间生成等间隔的8个数，不包含10，输出间隔
c=np.linspace(0,10,8,endpoint=False,retstep=True)
print(c)
```

### 方法四：ones方法创建（快速创建数值全为1的多维数组）

```
numpy.ones(shape, dtype=None, order='C')
```

- `shape`：用于指定数组形状。(a,b)代表a行b列。
- `dtype`：数据类型。
- `order`：按行或列方式储存数组。

```
d=np.ones((4,4))
print(d)#创建了一个4行4列的数组，数值全为1
```

### 方法五：zeros方法创建(快速创建数值全为0的多维数组)

和ones方法类似，不同之处在于数值全为0

```
e=np.zeros((2,3))
print(e)
```

### 方法六：eye方法创建（创建单位矩阵）

```
numpy.eye(N, M=None, k=0, dtype=<type 'float'>)
```

- `N`：输出数组的行数。
- `M`：输出数组的列数。
- `k`：对角线索引：0（默认）是指主对角线，正值是指上对角线，负值是指下对角线。

```
np.eye(5,4)#对角线在上方，也就是第一个元素为1
np.eye(5,4)#对角线在下方，也就是最后一个元素为1
```

### 方法七：从已知数据创建

- `frombuffer（buffer）`：将缓冲区转换为 1 维数组。

- `fromfile（file，dtype，count，sep）`：从文本或二进制文件中构建多维数组。

- `fromfunction（function，shape）`：通过函数返回值来创建多维数组。

- `fromiter（iterable，dtype，count）`：从可迭代对象创建 1 维数组。

- `fromstring（string，dtype，count，sep）`：从字符串中创建 1 维数组。

  例：

  ```
  np.fromfunction(lambda a,b:(a+1)*(b+1),(4,4))
  ```

  ## ndarray数组属性

  `ndarray.T`数组的转置

  `ndarray.dtype`输出数组包含元素的数据类型

  `ndarray.imag`输出数组包含元素的虚部

  `ndarray.real`输出数组包含元素的实部

  `ndarray.size`输出数组中的总包含元素数

  `ndarray.itemsize`输出一个数组元素的字节数

  `ndarray.nbypes`输出数组元素总字节数

  `ndarray.ndim`输出数组维度

  `ndarray.shape`输出数组形状

  `ndarray.strides`用来遍历数组时，输出每个维度中步进的字节数组（每隔多少个数跳一下）

  [更好的理解strides]: https://zhuanlan.zhihu.com/p/30960190

  ## 数组基本操作
  
  ### 重设形状reshape
  
  ```
  numpy.reshape(a, newshape)
  ```
  
  - a：原数组
  - newshape：指定新的形状（元素总数要相同，不然会报错）

```
a=np.arange(10).reshape((2,5))
```

### 数据展开ravel(将任意形状的数组扁平化，变成一维数组)

```
numpy.ravel(a, order='C')
```

- a：需要处理的数组

- order:变换时的读取顺序，默认是按照行依次读取，当order='F'时，按列读取。

  ### 轴移动moveaxis(将数组的轴移动到新的位置)

  ```
  numpy.moveaxis(a, source, destination)
  ```

  - `a`：数组。

  - `source`：要移动的轴的原始位置。

  - `destination`：要移动的轴的目标位置。

    移动之后，如果是二维，就是行和列互换；如果是三维，就是shape中索引source移动到destination，其他位置往前或往后挪。

    ### 轴交换swapaxes

    ```
    numpy.swapaxes(a, axis1, axis2)
    ```

    - `a`：数组。
    - `axis1`：需要交换的轴 1 位置。
    - `axis2`：需要与轴 1 交换位置的轴 1 位置。

```
a = np.ones((1, 4, 3))
a.shape, np.swapaxes(a, 0, 2).shape
((1, 4, 3), (3, 4, 1))
#索引0与索引2互换位置
```

### 数组转置transpose

```
numpy.transpose(a, axes=None)
```

- `a`：数组。
- `axis`：该值默认为 `none`，表示转置。如果有值，那么则按照值替换轴

**transpose 的本质，其实就是对 strides 中各个数的顺序进行调换**

像矩阵的转置，但可以选维度

```
a=np.arange(6).reshape(2,3)
print(a,np.transpose(a))
[[0 1 2]
 [3 4 5]] #a
 [[0 3]
 [1 4]
 [2 5]]#np.transpose(a)
```

### 维度改变atleast_xd(支持将输入数据的维度视为x维（x可以为1，2，3）)

```
numpy.atleast_xd()
```

```
print(np.atleast_3d([1,2,3]))
```

### 类型转换

- `asarray(a，dtype，order)`：将特定输入转换为数组array(一维数组)。

- `asanyarray(a，dtype，order)`：将特定输入转换为 ndarray（多维数组）。

- `asmatrix(data，dtype)`：将特定输入转换为矩阵。

- `asfarray(a，dtype)`：将特定输入转换为 float 类型的数组。

- `asarray_chkfinite(a，dtype，order)`：将特定输入转换为数组，检查 NaN 或 infs。

- `asscalar(a)`：将大小为 1 的数组转换为标量。

  ```
  a = np.arange(4).reshape(2, 2)
  np.asmatrix(a)  # 将二维数组转化为矩阵类型
  ```

  数组和矩阵长得很像，但数据类型不一样。

  [数组和矩阵应用上的区别]: https://zhuanlan.zhihu.com/p/163281949

  

### 数组连接concatenate(将多个数组沿指定轴连接在一起)

```
numpy.concatenate((a1, a2, ...), axis=0)
```

- `(a1, a2, ...)`：需要连接的数组。

- `axis`：指定连接轴。

  ```
  a=np.array([[1,2],[3,4],[5,6]])
  b=np.array([[7,8],[9,10]])
  c=np.array([[11,12]])
  print(np.concatenate((a,b,c)))#纵轴连接拼接在下面
  d=np.array([[7,8,9]])
  print(np.concatenate((a,d.T),axis=1))#横轴连接拼接在旁边（要求：保证连接处维数一致）
  ```
  
  ```
  [[ 1  2]
   [ 3  4]
   [ 5  6]
   [ 7  8]
   [ 9 10]
   [11 12]]#
  [[1 2 7]
   [3 4 8]
   [5 6 9]]#
  ```
  
  ### 数组堆叠stack
  
  **堆叠和连接是不一样的，连接是在维度对应时连成一个，堆叠就只是直接堆在一起**

- `stack(arrays，axis)`：沿着新轴连接数组的序列。
- `column_stack()`：将 1 维数组作为列堆叠到 2 维数组中。
- `hstack()`：按水平方向堆叠数组。
- `vstack()`：按垂直方向堆叠数组。
- `dstack()`：按深度方向堆叠数组。

```
a=np.array([[1,2],[3,4],[5,6]])
b=np.array([[7,8],[9,10],[11,12]])
print(np.stack((a,b)))
print(np.column_stack((a,b)))
print(np.hstack((a,b)))
print(np.vstack((a,b)))
print(np.dstack((a,b)))
```

```
[[[ 1  2]
  [ 3  4]
  [ 5  6]]

 [[ 7  8]
  [ 9 10]
  [11 12]]]#np.stack简单堆叠
[[ 1  2  7  8]
 [ 3  4  9 10]
 [ 5  6 11 12]]#np.column_stack
[[ 1  2  7  8]
 [ 3  4  9 10]
 [ 5  6 11 12]]#np.hstack水平
[[ 1  2]
 [ 3  4]
 [ 5  6]
 [ 7  8]
 [ 9 10]
 [11 12]]#np.vstack垂直
[[[ 1  7]
  [ 2  8]]

 [[ 3  9]
  [ 4 10]]

 [[ 5 11]
  [ 6 12]]]#np.dstack深度
```

### 数组拆分spilt

- `split(ary，indices_or_sections，axis)`：将数组拆分为多个子数组。

- `dsplit(ary，indices_or_sections)`：按深度方向将数组拆分成多个子数组。

- `hsplit(ary，indices_or_sections)`：按水平方向将数组拆分成多个子数组。

- `vsplit(ary，indices_or_sections)`：按垂直方向将数组拆分成多个子数组。

  二维数组可以按行分，按列分可以用转置

  ### 数组删除delete

  ```
  delete(arr，obj，axis)：沿特定轴删除数组中的子数组。
  ```

  - arr:数组
  - obj:索引
  - axis:0为横轴，1为列轴

```
a=np.arange(12).reshape(3,4)
np.delete(a,2,axis=1)#删除a数组中第三列
```

### 数组插入insert

```
insert(arr，obj，values，axis)：依据索引在特定轴之前插入值。
```

- values：需要插入的数组对象

### 数组附加append

```
append(arr，values，axis)：将值附加到数组的末尾，并返回 1 维数组。
```

相当于在末尾插入的insert，**返回值默认是展平的一维数组**

### 重设尺寸resize

```
resize(a，new_shape)：对数组尺寸进行重新设定。
```

`resize` 和 `reshape` 一样，都是改变数组原有的形状。区别在于**是否影响原数组**。`reshape` 在改变形状时，不会影响原数组，相当于对原数组做了一份拷贝，而 `resize` 则是对原数组执行操作。

### 翻转数组

- `fliplr(m)`：左右翻转数组。

- `flipud(m)`：上下翻转数组。

  （关于对称轴对称）

### 数组索引和切片

#### 数组索引（从0开始）

##### 一维数组索引：和list一致

```
a=np.arange(10)
print(a[1])#取一个
print(a[[1,2,3]])#取多个，多加一层[]，用逗号隔开
```

##### 二维数组索引：

```
a=np.arange(20).reshape(4,5)
print(a[1,2])#取第二行第三列的元素，不加[],直接用逗号隔开（取一个）
print(a[[1,2],[3,4]])#用逗号把[]隔开(取多个)
```

⚠️tips：这里需要注意索引的对应关系。我们实际获取的是 `[1, 3]`，也就是第 `2` 行和第 `4` 列对于的值 `8`。以及 `[2, 4]`，也就是第 `3` 行和第 `5` 列对应的值 `14`。

多一层维度就加一级`[]`，使用`，`分割

#### 数组切片

```
Ndarray[start:stop:step]#[起始索引：截至索引：步长]
```

##### 一维数组切片

空着就默认为初始的开始或结束，截至索引是不包含它本身的

```
a=np.arange(10)
print(a)
print(a[1:])
print(a[:-1])
print(a[:5])
print(a[:8:2])
```

```
[0 1 2 3 4 5 6 7 8 9]
[1 2 3 4 5 6 7 8 9]
[0 1 2 3 4 5 6 7 8]
[0 1 2 3 4]
[0 2 4 6]
```

##### 多维数组切片（用逗号分隔不同维度）

```
b=np.arange(20).reshape(4,5)
print(b)
print(b[0:3,2:4])
print(b[:,::2])
```

[0:3,2:4]中0：3就是第一个维度（也就是行）取第一行到第三行；2：4就是第二个维度（在这里是列）取第三列到第四列，得到一个新数组。

[:,::2]指的是按步长为2取所有列和行的数据

### 排序numpy.sort

```
numpy.sort(a, axis=-1, kind='quicksort', order=None)
```

- `a`：数组。

- `axis`：**要排序的轴**。如果为 `None`，则在排序之前将数组铺平。默认值为 `-1`，沿最后一个轴排序。

  在二维数组里面axis=0是按列排序，axis=1或-1是按行排序

- `kind`：`{'quicksort'，'mergesort'，'heapsort'}`，排序算法。默认值为 `quicksort`。

#### 对数组排序的其他方法：

- `numpy.lexsort(keys ,axis)`：使用多个键进行间接排序。
- `numpy.argsort(a ,axis,kind,order)`：沿给定轴执行间接排序。（出现顺序）
- `numpy.msort(a)`：沿第 1 个轴排序。（也就是按列排序）
- `numpy.sort_complex(a)`：针对复数排序。

### 搜索和计数

- `argmax(a ,axis,out)`：返回数组中指定轴的最大值的索引。

- `nanargmax(a ,axis)`：返回数组中指定轴的最大值的索引,忽略 NaN（not a number非数，未定义或不可表示的值）。

- `argmin(a ,axis,out)`：返回数组中指定轴的最小值的索引。

- `nanargmin(a ,axis)`：返回数组中指定轴的最小值的索引,忽略 NaN。

- `argwhere(a)`：返回数组中非 0 元素的索引,按元素分组。

- `nonzero(a)`：返回数组中非 0 元素的索引。

- `flatnonzero(a)`：返回数组中非 0 元素的索引,并铺平。

- `where(条件,x,y)`：根据指定条件,从指定行、列返回元素。

  [（a>0,a,2)条件a>0，成立返回a，不成立返回2]

- `searchsorted(a,v ,side,sorter)`：查找要插入元素以维持顺序的索引。

- `extract(condition,arr)`：返回满足某些条件的数组的元素。

- `count_nonzero(a)`：计算数组中非 0 元素的数量。

  ## Numpy随机数

  ### random.rand(d0,d1,...,dn)

  指定一个给定维度的数组((d0,d1,...,dn)就是维度)并使用[0，1)区间随机数据填充，数据**均匀分布**。

```
np.random.rand(2,5)
```

### random.randn(d0,d1,...,dn)

与上一个的区别在于：`randn`是从**标准正态分布**中返回一个或多个样本值，而`rand`是均匀分布。

### random.randint(low,high,size,dtype)

会生成[low,high)的随机整数，半开半闭

```
np.random.randint(2,5,10)#从[2,5)之间生成10个随机数
```

### random.random_sample(size)

会在[0,1)区间生成指定size的随机浮点数

size就是生成随机数的个数

与 `numpy.random.random_sample` 类似的方法还有：

- `numpy.random.random([size])`
- `numpy.random.ranf([size])`
- `numpy.random.sample([size])`

### random.choice(a,size,replace,p)

从给定的数组里随机抽取几个值（随机取样）

```
np.random.choice(10, 5) # 在 0~9 中随机抽取5个数
```

### 其他（需要再学）

![image-20221101203601126](C:\Users\bamboo\AppData\Roaming\Typora\typora-user-images\image-20221101203601126.png)

![image-20221101203623765](C:\Users\bamboo\AppData\Roaming\Typora\typora-user-images\image-20221101203623765.png)

![image-20221101203649021](C:\Users\bamboo\AppData\Roaming\Typora\typora-user-images\image-20221101203649021.png)

## 数学函数

### 三角函数

- `numpy.sin(x)`：三角正弦。
- `numpy.cos(x)`：三角余弦。
- `numpy.tan(x)`：三角正切。
- `numpy.arcsin(x)`：三角反正弦。
- `numpy.arccos(x)`：三角反余弦。
- `numpy.arctan(x)`：三角反正切。
- `numpy.hypot(x1,x2)`：直角三角形求斜边。
- `numpy.degrees(x)`：弧度转换为度。
- `numpy.radians(x)`：度转换为弧度。
- `numpy.deg2rad(x)`：度转换为弧度。
- `numpy.rad2deg(x)`：弧度转换为度。

### 双曲函数（经常出现于某些重要的线性微分方程的解中）

- `numpy.sinh(x)`：双曲正弦。
- `numpy.cosh(x)`：双曲余弦。
- `numpy.tanh(x)`：双曲正切。
- `numpy.arcsinh(x)`：反双曲正弦。
- `numpy.arccosh(x)`：反双曲余弦。
- `numpy.arctanh(x)`：反双曲正切。

### 数值修约（统一位数，舍去尾数）

在进行具体的数字运算前，按照一定的规则确定一致的位数，然后舍去某些数字后面多余的尾数的过程。

- `numpy.around(a)`：平均到给定的小数位数。
- `numpy.round_(a)`：将数组舍入到给定的小数位数。
- `numpy.rint(x)`：修约到最接近的整数。
- `numpy.fix(x, y)`：向 0 舍入到最接近的整数。
- `numpy.floor(x)`：返回输入的底部(标量 x 的底部是最大的整数 i)。
- `numpy.ceil(x)`：返回输入的上限(标量 x 的底部是最小的整数 i).
- `numpy.trunc(x)`：返回输入的截断值。

### 求和、求积、差分

- `numpy.prod(a, axis, dtype, keepdims)`：返回指定轴上的数组元素的**乘积**。

- `numpy.sum(a, axis, dtype, keepdims)`：返回指定轴上的数组元素的**总和**。

- `numpy.nanprod(a, axis, dtype, keepdims)`：返回指定轴上的数组元素的乘积, 将 NaN 视作 1。

- `numpy.nansum(a, axis, dtype, keepdims)`：返回指定轴上的数组元素的总和, 将 NaN 视作 0。

- `numpy.cumprod(a, axis, dtype)`：返回沿给定轴的元素的累积乘积。

- `numpy.cumsum(a, axis, dtype)`：返回沿给定轴的元素的累积总和。

- `numpy.nancumprod(a, axis, dtype)`：返回沿给定轴的元素的累积乘积, 将 NaN 视作 1。

- `numpy.nancumsum(a, axis, dtype)`：返回沿给定轴的元素的累积总和, 将 NaN 视作 0。

- `numpy.diff(a, n, axis)`：计算沿指定轴的第 n 个离散差分。

- `numpy.ediff1d(ary, to_end, to_begin)`：数组的连续元素之间的差异。

- `numpy.gradient(f)`：返回 N 维数组的梯度。

- `numpy.cross(a, b, axisa, axisb, axisc, axis)`：返回两个(数组）向量的叉积。

- `numpy.trapz(y, x, dx, axis)`：使用复合梯形规则沿给定轴积分。

  ##### 加法函数np.sum()和np.cumsum()区别

np.sum()是直接整体或整行或整列求和，而np.cumsum是每个元素都带有之前的累加值，一一累加。

```
np.sum(a)#整体求和
np.sum(a,axis=0)#对纵轴进行操作
np.sum(a,axis=1)#对横轴进行操作
```

```
x = np.array([[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]])
y = np.cumsum(x)
print(y)#一一累加
[ 11  23  36  50  65  81  98 116 135 155 176 198 221 245 270]

y = np.cumsum(x,axis =0)
print(y)#按列一一累加
[[11 12 13 14 15]
 [27 29 31 33 35]
 [48 51 54 57 60]]

y = np.cumsum(x,axis =1)
print(y)#按行一一累加
[[ 11  23  36  50  65]
 [ 16  33  51  70  90]
 [ 21  43  66  90 115]] 

```

### 指数和对数

- `numpy.exp(x)`：计算输入数组中所有元素的指数。（e的x次方）
- `numpy.log(x)`：计算自然对数。（lnx）
- `numpy.log10(x)`：计算常用对数。（lgx）
- `numpy.log2(x)`：计算二进制对数。（log2x）

### 算术运算（可以直接针对数组）

- `numpy.add(x1,x2)`:对应元素相加。(x1、x2可以都是数组，如果x1是一行，那x2是多行，则x1与x2每一行相加，或x1、x2shape相同)
- `numpy.reciprocal(x)`：求倒数 1/x。（默认输出是整数数组）
- `numpy.negative(x)`：求对应负数。
- `numpy.multiply(x1, x2)`：求解乘法。（元素里面的数相乘）
- `numpy.divide(x1, x2)`：相除 x1/x2。
- `numpy.power(x1, x2)`：类似于 x1^x2。
- `numpy.subtract(x1, x2)`：减法。
- `numpy.fmod(x1, x2)`：返回除法的元素余项。
- `numpy.mod(x1, x2)`：返回余项。
- `numpy.modf(x1)`：返回数组的小数和整数部分。
- `numpy.remainder(x1, x2)`：返回除法余数。

### 矩阵和向量积

- `numpy.dot(a, b)`：求解两个数组的点积。
- `numpy.vdot(a, b)`：求解两个向量的点积。
- `numpy.inner(a, b)`：求解两个数组的内积。
- `numpy.outer(a, b)`：求解两个向量的外积。
- `numpy.matmul(a, b)`：求解两个数组的矩阵乘积。
- `numpy.tensordot(a, b)`：求解张量点积。
- `numpy.kron(a, b)`：计算 Kronecker 乘积。

### 其他数学运算

- `numpy.angle(z, deg)`：返回复参数的角度。
- `numpy.real(val)`：返回数组元素的实部。
- `numpy.imag(val)`：返回数组元素的虚部。
- `numpy.conj(x)`：按元素方式返回共轭复数。
- `numpy.convolve(a, v, mode)`：返回线性卷积。
- `numpy.sqrt(x)`：平方根。
- `numpy.cbrt(x)`：立方根。
- `numpy.square(x)`：平方。
- `numpy.absolute(x)`：绝对值, 可求解复数。
- `numpy.fabs(x)`：绝对值。
- `numpy.sign(x)`：符号函数
- `numpy.maximum(x1, x2)`：最大值。
- `numpy.minimum(x1, x2)`：最小值。
- `numpy.nan_to_num(x)`：用 0 替换 NaN。
- `numpy.interp(x, xp, fp, left, right, period)：`线性插值。

### 代数运算

- `numpy.linalg.cholesky(a)`：Cholesky 分解。
- `numpy.linalg.qr(a ,mode)`：计算矩阵的 QR 因式分解。
- `numpy.linalg.svd(a ,full_matrices,compute_uv)`：奇异值分解。
- `numpy.linalg.eig(a)`：计算正方形数组的特征值和右特征向量。
- `numpy.linalg.eigh(a, UPLO)`：返回 Hermitian 或对称矩阵的特征值和特征向量。
- `numpy.linalg.eigvals(a)`：计算矩阵的特征值。
- `numpy.linalg.eigvalsh(a, UPLO)`：计算 Hermitian 或真实对称矩阵的特征值。
- `numpy.linalg.norm(x ,ord,axis,keepdims)`：计算矩阵或向量范数。
- `numpy.linalg.cond(x ,p)`：计算矩阵的条件数。
- `numpy.linalg.det(a)`：计算数组的行列式。
- `numpy.linalg.matrix_rank(M ,tol)`：使用奇异值分解方法返回秩。
- `numpy.linalg.slogdet(a)`：计算数组的行列式的符号和自然对数。
- `numpy.trace(a ,offset,axis1,axis2,dtype,out)`：沿数组的对角线返回总和。
- `numpy.linalg.solve(a, b)`：求解线性矩阵方程或线性标量方程组。
- `numpy.linalg.tensorsolve(a, b ,axes)`：为 x 解出张量方程 a x = b
- `numpy.linalg.lstsq(a, b ,rcond)`：将最小二乘解返回到线性矩阵方程。
- `numpy.linalg.inv(a)`：计算逆矩阵。（必须是方阵，且行列式不等于0）
- `numpy.linalg.pinv(a ,rcond)`：计算矩阵的（Moore-Penrose）伪逆。
- `numpy.linalg.tensorinv(a ,ind)`：计算 N 维数组的逆。
