## 背包问题

### 0/1 背包

- $\text{N}$件物品，容量$\text{V}$背包，每件物品只能使用一次

```python
# 二维动态规划写法
n, v = map(int, input().split())
w = [0]
p = [0]
for i in range(n): 
    a, b = map(int, input().split())
    w.append(a)
    p.append(b)
# 状态转移方程为 $f(i,mx)=max(f(i-1,mx),f(i-1,mx-w_i)+p_i)$
f=[[0] * (v+1) for i in range(n+1)]
for i in range(1, n+1):
    for j in range(v+1):
        # 直接不动
        f[i][j] = f[i-1][j]
        if j >= w[i]:
            f[i][j] = max(f[i-1][j], f[i-1][j-w[i]] + p[i])
print(f[n][v])
```

```python
# 一维动态规划的解法
# 一维动态规划写法
n, v = map(int, input().split())
w = [0]
p = [0]
for i in range(n):
    a, b = map(int, input().split())
    w.append(a)
    p.append(b)
# 状态转移方程为 $f(i,mx)=max(f(i-1,mx),f(i-1,mx-w_i)+p_i)$
f = [0] * (v+1)
for i in range(1, n+1):
    for j in range(v, w[i]-1, -1):
        # 为什么这部分需要倒着进行更新
        # 因为再下一次进行更新的时候总是用的前面的状态
        # 而且进行当前状态更新的时候也是需要用到前面的状态
        # 所以更新要从后面开始
        f[j] = max(f[j], f[j-w[i]] + p[i])
print(f[v])
```

### 完全背包

- $\text{N}$件物品，容量$\text{V}$背包，每件物品无限使用

```python
n, v = map(int, input().split())
P = []
W = []
for _ in range(n):
    w, p = map(int, input().split())
    P.append(p)
    W.append(w)
f = [[0] * (v + 1) for _ in range(n + 1)]
for i in range(1, n + 1):
    for j in range(1, v + 1):
        f[i][j] = f[i - 1][j]
        if j >= W[i - 1]:
            f[i][j] = max(f[i][j], f[i][j - W[i - 1]] + P[i - 1])
print(f[n][v])
```

### 分组背包

## 换根DP

```python
def sumOfDistancesInTree(self, n: int, edges: List[List[int]]) -> List[int]:
    # ans[0] 已经计算出来了
    # ans[2] = ans[0] + n - size[2] - size[2]
    g = [[] for _ in range(n)]
    for x, y in edges:
        g[x].append(y)
        g[y].append(x)
    ans = [0] * n
    size = [1] * n
    def dfs(x, fa, depth):
        ans[0] += depth
        for y in g[x]:
            if y != fa:
                dfs(y, x, depth + 1)
                size[x] += size[y] 
    dfs(0, -1, 0)
    def reroot(x, fa):
        for y in g[x]:
            if y != fa:
                ans[y] = ans[x] + n - 2 * size[y]
                reroot(y, x)
    reroot(0, -1)
    return ans
```

## 数位DP

```python
def countSpecialNumbers(n: int) -> int:
        s = str(n)
        # i代表当前填到了第几位数
        # mask类似于状态压缩的思想，看看前面是否已经使用过了当前的数字
        # is_lim 代表前面的数字是否都填了对应位置的数字，如果为True,接下来只能填 0-s[i],否则为0-9
        # is_num 代表前面是否填了数字，如果填了数字，接下来这这个可以从0开始,否则从1开始
        @cache
        def f(i: int, mask: int, is_lim: bool, is_num: bool):
            res = 0
            if i == len(s):
                return int(is_num)#填了返回1,没填直接返回0
            if not is_num:#如果没有填数字,可以继续选择跳过
                res += f(i+1, mask, False, is_num)
            up = int(s[i]) if is_lim else 9 #如果前面都对号入座了,则接下来填的数字最大值就是s[i]
            for d in range(1-int(is_num), up+1):
                if mask>>d&1 == 0:
                    res += f(i+1, mask|1<<d, is_lim and d==int(s[i]), True)
            return res
        return f(0, 0, True, False)
```

## 线性DP

- 最长上升子序列二分做法

```python
def lengthOfLIS(self, nums: List[int]) -> int:
        # 二分做法
        # 每次的数都要比当前的大即可
        n = len(nums)
        q = []
        for i in range(n):
            l, r = 0, len(q)-1
            while l < r:
                mid = (l + r) // 2
                if q[mid] >= nums[i]:
                    r = mid
                else:
                    l = mid + 1
            if len(q) == 0 or q[l] < nums[i]:
                q.append(nums[i])
            else:
                q[l] = nums[i]
        return len(q)
```

## 状态压缩DP

## 子集DP
