## 逆元模板

```python
import sys
input = lambda: sys.stdin.readline().rstrip("\r\n")

MOD = 1000000007
n = int(input())
MX = 100010
fac = [0] * MX
fac[0] = 1
for i in range(1, MX):
    fac[i] = fac[i - 1] * i % MOD

inv_fac = [0] * MX
inv_fac[MX - 1] = pow(fac[MX - 1], MOD - 2, MOD)
for i in range(MX - 1, 0, -1):
    inv_fac[i - 1] = inv_fac[i] * i % MOD

def comb(n: int, k: int) -> int:
    return fac[n] * inv_fac[k] % MOD * inv_fac[n - k] % MOD

def solve():
    a, b = map(int, input().split())
    print(comb(a, b))
    
for _ in range(n):
    solve()
```

## 高斯消元法求解线性方程组

```python
n = int(input())
eps = 1e-6
a = []
def out():
    for i in range(n):
        for j in range(n+1):
            print(a[i][j], end = ' ')
        print()
    print() 
def guass():
    r, c = 0, 0
    # 枚举每一列
    while c < n:
        # 找到该列中 行最大值
        t = r
        for i in range(r,n):
            if abs(a[t][c]) < abs(a[i][c]):
                t = i
        # 如果该列最大值为零,可以直接跳过
        if abs(a[t][c]) < eps:
            c += 1
            continue
        # 进行交换
        for i in range(n+1):
            a[t][i], a[r][i] = a[r][i], a[t][i]
        
        # 归一
        for i in range(n, c-1, -1):
            a[r][i] /= a[r][c]
        
        # 消去剩下列的值
        for i in range(r+1, n):
            for j in range(n, c-1, -1):
                a[i][j] -= a[i][c]*a[r][j]
        r += 1
        c += 1
        # out()
    # 回代
    for i in range(n-1, -1, -1):
        for j in range(i+1, n):
            a[i][n] -= a[i][j]*a[j][n]
    if r < n:
        i = r
        while i < n:
            if abs(a[i][n]) > eps:
                return 2
            i += 1
        return 1
    return 0 
    
for i in range(n):
    a.append(list(map(float, input().split())))

k = guass()
if k == 0:
    for i in range(n):
        if abs(a[i][n]) < eps:
            a[i][n] = 0
        print("%.2f"%a[i][n])
elif k == 1:
    print("Infinite group solutions")
else:
    print("No solution")

```

## 快速幂

```python
def qmi(a, k, p):
    res = 1 
    while k:
        if k & 1:
            res = res * a % p
        a *= a%p
        k >>= 1
    return res
```

## 扩展欧几里得

- 给定 $n$ 组数据 $a_i, b_i, m_i$ ,对于每组数求出一个 $x_i$ ,使其满足 $a_i \times x_i = b_i (mod ~ m_i)$
  - 即求解 $ax + my = b$ ,需要保证 $(a, m) | b$ 否则无解

```python
n = int(input())

def exgcd(a, b):
    if b == 0:
        return a, 1, 0
    d, _x, _y = exgcd(b, a%b)
    x = _y
    y = _x - a // b * _y
    return d, x, y

for i in range(n):
    a, b, m = map(int, input().split())
    d, x, y = exgcd(a, m)
    if b % d:
        print('impossible')
    else:
        print(x * (b // d) % m)
```

# 约数

## 试除法求约数

```python
n = int(input())
def ff(x):
    pre = []
    las = []
    i = 1
    while i <= x // i:
        if x % i == 0:
            pre.append(i)
            if i != x // i:
                las.append(x//i)
        i += 1
    las.reverse()
    print(' '.join(str(i) for i in pre), end=' ')
    print(' '.join(str(i) for i in las))
for i in range(n):
    x = int(input())
    ff(x)

```

## 约数的个数

- 公式:
  - $\large x=p_1^{\alpha_1}p_2^{\alpha_2}p_3^{\alpha_3}\dots$
  - 所以约数的个数就是 $(1+\alpha_1)(1+\alpha_2)(1+\alpha_3)\dots$

```python
import collections
from collections import defaultdict
n = int(input())
dic = defaultdict(int)#存放所有质因子的个数即可
MOD = 1e9 + 7
def ff(x):
    i = 2
    while i <= x // i:
        cnt = 0
        if x % i == 0:
            while x % i == 0:
                x //= i
                cnt += 1
            dic[i] += cnt
        i += 1
    if x > 1:
        dic[x] += 1

for i in range(n):
    x = int(input())
    ff(x)
res = 1
for x, y in dic.items():
    res = res * (y+1) % MOD
print(int(res))
```

## 约数之和

- 公式
  - $\large (p_1^0+p_1^1+\dots+p_1^{\alpha_1})(p_2^0+p_2^1+\dots+p_2^{\alpha_2})\dots$  

```python
import collections
from collections import defaultdict
n = int(input())
dic = defaultdict(int)
MOD = int(1e9+7)
def ff(x):
    i = 2
    while i <= x // i:
        cnt = 0
        if x % i == 0:
            while x % i == 0:
                x //= i
                cnt += 1
            dic[i] += cnt
        i += 1
    if x > 1:
        dic[x] += 1
for i in range(n):
    x = int(input())
    ff(x)
res = 1
for q, k in dic.items():
    res = res * (((q**(k+1)-1) // (q-1)) % MOD) % MOD
        
print(int(res) % MOD)
```

## 最大公约数

```python
n = int(input())
def gcd(x, y):
    return x if y==0 else gcd(y, x%y)
for i in range(n):
    a, b = map(int, input().split())
    print(gcd(a, b))
```

# 质数

- 定义:在大于1的整数中,如果只包含 1 和本身这两个约数,就被称为质数
- 质数的判定:试除法

## 试除法判定质数

```python
import math
n = int(input())
def is_prime(x):
    if x < 2:
        return False
    up = int(math.sqrt(x)) + 1
    for i in range(2, up):
        if x % i == 0:
            return False
    return True

for i in range(n):
    x = int(input())
    if is_prime(x):
        print('Yes')
    else:
        print('No')
```

- 时间复杂度 $O(\sqrt{n})$

## 分解质因数

- 从小到大枚举 $n$ 的所有因数
- $n$ 中只包含一个大于 $\sqrt{n}$ 的质因子

```python
import math
n=int(input())

def divide(x):
    i = 2
    while i <= x // i:
        t = 0
        if x % i == 0:
            while x % i == 0:
                t += 1
                x //= i
            print(i, t)
        i += 1
    if x > 1:
        print(x, 1)
    print()
            
for i in range(n):
    a=int(input())
    divide(a)
   
```

## 筛质数

- 即判断 1-n 中存在多少质数

### 埃式筛法

- 每次都筛除最小质因子的倍数

```python
# 普通筛
vis = [False] * (n+1)
prime = []
## 求不超过n的质数
for i in range(2, n+1):
    if not vis[i]:
        prime.append(i)
        cnt += 1
        # 通过质数来进行筛选,这样就可以不进行重复的计算,比如4,之前已经被2 筛掉了
        # 但是这样依旧会有重复比如,6会被2和3重复筛两次
        j = i
        while j <= n:
            vis[j] = True
            j += i
```

```c++
const int MX = 1e6;
vector<int> primes;
bool p[MX + 1];
for (int i = 2; i <= n; i++) {
    if(!p[i]) continue; 
    primes.push_back(i);
    for (int j = i; j <= n / i; j++) // 避免溢出的写法
        p[i * j] = false;
}
```

- 时间复杂度为 $O(nlnlnn)$

### 线性筛法

- 算法思想就是每一个数都通过自己最小的质因子筛除,比如 6只能由2来筛除

```python
n = int(input())
cnt = 0
vis = [False] * (n+1)
prime = []
#1、首先枚举2-n的数
for i in range(2, n+1):
    # 如果当前没有被筛掉，那一定是质数
    if not vis[i]:
        prime.append(i)
        cnt += 1
    # 枚举从小到大的质数
    j = 0
    # 要保证 prime[j]*i<=n
    # 这里是不会越界的,因为最后一个数如果是质数,则到最后一步break
    # 如果是合数,则将会被提前break
    while prime[j] <= n // i:
        # 通过prime[j]*i 将这个数字筛掉,可以推理得出,prime[j] 一定是prime[j]*i的最小质因子
        # 如果i%prime[j]==0 那么上一步筛的prime[j] 一定是i的最小质因子,并且不存在比prime[j]*i 所有因子还要小的质因子
        # 如果i%prime[j]!=0 那么上一步筛的prime[j] 一定是prime[j]*i的最小质因子
        # 如果i%prime[j]!=0,那么将继续筛,下一步就会继续如此操作
        vis[prime[j] * i] = True
        if i % prime[j] == 0: break
        j += 1
```

## 中国剩余定理

$\begin{cases}
x~mod~a_1 \equiv m_1 \\
x~mod~a_2 \equiv m_2  \\
\qquad \vdots \\
x~mod~a_n \equiv m_n
\end{cases}$
求满足条件的最小 $x$ 解

- 证明过程:
  - 首先计算两个方程的解 $x~mod~a_1 \equiv m_1,x~mod~a_2 \equiv m_2$,令 $x=k_1 a_1+m_1,x=k_2a_2+m_2$
  $\rightarrow k_1a_1+m_1=k_2a_2+m_2$
  $\rightarrow k_1a_1-k_2a_2=m_2-m_1$
  需要满足的条件是
  $\rightarrow (a_1,a_2)|(m_2-m_1)$
  可以通过扩展欧几里得算法求解 $k_1,k_2$ 的值
  令:
  $\large \begin{cases}
  k_1=k_1+k \frac{a_2}{d} \\
  k_2=k_2+k \frac{a_1}{d}
  \end{cases}$
  则 $x$ 的通解为 $x=(k_1+k \frac{a_2}{d})a_1+m_1$
  $\large \rightarrow x=a_1k_1+m_1+k \frac{a_1a_2}{d}$
  因此就可以将两个同余方程转换成一个同余方程 $x=x_0+k \alpha , \alpha=[a_1,a_2]$ 最小公倍数

- 中国剩余定理的通解是:
  - $x=\sum_{i=1}^{n} a_iM_i*M_i^{-1}+kM,M=\prod_{i=1}^{n}M_i$

```python

def exgcd(a, b):
    if b == 0:
        return a, 1, 0
    d, _x, _y=exgcd(b, a%b)
    x = _y
    y = _x - a // b * _y
    return d, x, y

n = int(input())
a1, m1 = map(int, input().split())
flag = True
for i in range(n-1):
    a2, m2 = map(int, input().split())
    d, k1, k2 = exgcd(a1, a2)
    if (m2-m1) % d:
        print(-1)
        flag = False
        break
    k1 *= (m2 - m1) // d
    t = a2 // d
    k1 = (k1 % t + t) % t
    m1 = a1 * k1 + m1
    a1 = a1 // d * a2
if flag:
    print((m1 % a1 + a1) % a1)

```

# 欧拉函数

- 即求解 $1 \sim N$ 中间和 $N$ 互质的数的个数

## 欧拉函数

- 利用欧拉函数的公式求解即可
  - $\large \phi(N) = N * \frac{p_1-1}{p_1} \frac{p_2-1}{p_2} \dots \frac{p_k-1}{p_k}$
  - $prove:$ 通过容斥原理证明:首先需要明白欧拉函数的含义,所以首先可以将 $N = p_1^{\alpha_1}\dots p_k^{\alpha_k}$
  然后用 $N$ 减去所有关于这些质因子组成的倍数,即最后的与 $N$ 互质的个数

```python
n = int(input())
def fe(x):
    i = 2
    res = x
    while i <= x // i:
        if x % i == 0:
            res = res // i * (i-1)
            while x % i == 0:
                x //= i
        i += 1
    if x > 1:
        res = res//x*(x-1)
    return res

for i in range(n):
    x = int(input())
    print(fe(x))

```

## 筛法求欧拉函数

- 使用线性筛法求 $1 \sim N$ 的欧拉函数的和

```python
n = int(input())
phi = [1] * (n+1)
prime = []
vis = [False] * (n+1)
def xxsf(n):
    # 线性筛法
    cnt = 0
    for i in range(2, n+1):
        if not vis[i]:
            phi[i] = i-1
            prime.append(i)
            cnt += 1
        j = 0
        while prime[j] <= n//i:
            vis[prime[j]*i] = True
            if i % prime[j] == 0:
                phi[i*prime[j]] = phi[i] * prime[j]
                break
            phi[i*prime[j]] = phi[i] * (prime[j]-1)
            j += 1
    return sum(phi[1:])
print(xxsf(n))
```

# 组合数

## 卢卡斯定理

- 适用情况是 $C_a^b$ 中 $a,b$ 都是比较大的情况
$\large a=a_0 p^0+a_1 p^1+\dots+a_k p^k$
$\large b=b_0 p^0+b_1 p^1+\dots+b_k p^k$
考虑生成函数
$\large (1+x)^a=(1+x)^{a_0 p^0+a_1 p^1+\dots+a_k p^k}$
$\large \rightarrow (1+x)^{a_0p^0}(1+x)^{a_1p^1}\dots (1+x)^{a_k p_k}$
另外有
$\large (1+x)^p~mod~p=(C_p^0 x^0+C_p^1 x^1+\dots+C_p^p x^p)~mod~p$
$\large =1+x^p$
因此
$\large \rightarrow (1+x)^{a_0p^0}(1+x)^{a_1p^1}\dots (1+x)^{a_k p_k}~mod~p$
$\large \rightarrow (1+x)^{a_0p^0}(1+x)^{a_1p^1}\dots (1+x)^{a_k p_k}~mod~p$
$\large \rightarrow (1+x^{p_0})^{a_0}(1+x^{p_1})^{a_1}\dots (1+x^{p_k})^{a_k} ~mod~p$
即
考虑生成函数
$\large (1+x)^a ~mod~p=(1+x^{p^0})^{a_0}(1+x^{p^1})^{a_1}\dots (1+x^{p^k})^{a_k} ~mod~p$
两边同时考虑 $x^b$ 这一项
则 $\large C_a^b x^b=C_{a_0}^{b_0}x^{b_0p^0} C_{a_1}^{b_1}x^{b_1p^1}\dots C_{a_k}^{b_k}x^{b_kp^k}=C_{a_0}^{b_0}C_{a_1}^{b_1}\dots C_{a_k}^{b_k} x^{b_0 p^0+b_1 p^1+\dots+b_k p^k}$
即
$\large C_a^b=C_{a_0}^{b_0}C_{a_1}^{b_1}\dots C_{a_k}^{b_k} $
可以进一步改写成
$\large C_a^b=C_{a \% p}^{b \% p}C_{a/p}^{b/p}~mod~p$
这是一个递归的形式

```python
MOD = int(1e9 + 7)
N = int(100003)
# 快速幂求乘法逆元
def qmi(a, k, p):
    res = 1
    while k:
        if k & 1:
            res = res*a % p
        a = a*a % p
        k >>= 1
    return res
fact = [1] * N
infact = [1] * N
def init():
    for i in range(1, N):
        fact[i] = (fact[i-1] * i) % MOD
        infact[i] = infact[i-1] * qmi(i, MOD-2, MOD) % MOD
init()
n = int(input())
for i in range(n):
    a, b = map(int, input().split())
    print(fact[a] * infact[b] % MOD * infact[a-b] % MOD)

```
