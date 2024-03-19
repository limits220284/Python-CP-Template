## floyed 算法

- 其实是动态规划的思想
  - $f(k,i,j)=min(f(k-1,i,j),f(k-1,i,k)+f(k-1,k,j))$
  - 思想其实就是将第$k$个点加入,然后通过第 $k$ 个点来更新 $i,j$ 两点之间的距离
  - 上述状态转移方程可以写成 $f(i,j)=min(f(i,j),f(i,k)+f(k,j))$ 是因为每一次用的都是上一次的状态图,因为要更新 $f(i,j)$ 的时候,会用到 $f(i,k),f(k,j)$ 这些都没有被更新过。会不会出现一种情况就是 $f(i,k)$ 在更新 $f(i,j)$ 之前就被更新过了? 答案是不可能的。
  - 可以考虑 $f(i,j)=min(f(i,j),f(i,k)+f(k,j))$
  有 $f(k-1,i,k)=min(f(k-1,i,k),f(k-1,i,k)+f(k-1,k,k))=f(k-1,i,k)$ 因为从 $k$ 点到 $k$ 点题目要求是没有负权回路的,这样一来虽然某种意义上算是对其进行了更新,但是更新后的值也是不变的。

```python
n, m, q = map(int, input().split())
mx = 1e9 + 7
g=[[mx] * (n+1) for i in range(n+1)]

def floyed():
    for k in range(1, n+1):
        for i in range(1, n+1):
            for j in range(1, n+1):
                g[i][j] = min(g[i][j], g[i][k] + g[k][j])

for i in range(1, n+1):
    g[i][i] = 0
for i in range(m):
    x, y, w = map(int, input().split())
    g[x][y] = min(g[x][y], w)

floyed()

for i in range(q):
    x, y = map(int, input().split())
    t = g[x][y]
    if t > mx / 2:
        print('impossible')
    else:
        print(t)

```

## 朴素版本的dijkstra算法

```python
n, m = map(int, input().split())
mx = float('inf')
g = [[mx] * (n+1) for i in range(n+1)]
vis = [False] * (n+1)
dis = [mx] * (n+1)
def dijkstra(be, ed):
    dis[be] = 0
    for i in range(1, n+1):
        # 首先找到一个边最小值
        t = -1
        for j in range(1, n+1):
            if not vis[j] and (t == -1 or dis[t] > dis[j]):
                t = j
        vis[t] = True
        # 更新起点到终点的权值
        for k in range(1, n+1):
            dis[k] = min(dis[k], dis[t] + g[t][k])
    return dis[ed]

for i in range(m):
    x, y, z = map(int, input().split())
    g[x][y] = min(g[x][y], z)

k = dijkstra(1, n)
if k == mx:
    print('-1')
else:
    print(k)

```

## 堆优化版本的 dijkstra 算法

- 适用于稀疏图

```python

import heapq

n, m = map(int, input().split())
g = [[] for _ in range(n+1)]
for i in range(m):
    a, b, w = map(int, input().split())
    g[a].append((b,w))
mx = 10**9 + 7
q = []
dis = [mx] * (n+1)
vis = [False] * (n+1)
heapq.heappush(q, (0, 1))
dis[1] = 0
while q:
    d, x = heapq.heappop(q)
    if vis[x]:
        continue
    vis[x] = True
    if x == n:
        break
    for y, w in g[x]:
        dis[y] = min(dis[y], w+d)
        heapq.heappush(q, (dis[y], y))

print(-1 if dis[n] == mx else dis[n])
```

## spfa求最短路

- 算法的主要思想是对于bellman_ford算法的改进
- 因为bellman_ford 算法是通过边松弛来做的, dis[b] = min(dis[b], dis[a] + w)
可以通过上面的式子看出,只有dis[a]在变小的时候,dis[b]才会变小。所以spfa会通过队列的方式先将变小的点存储起来，然后通过这些点来更新与其相邻的点。
- spfa也侧面反映了，bellman_ford算法存在的不足，后者在更新的过程中，如果dis[a]本身就已经不变了，那么再通过 $a \leftrightarrow b$ 之间的边来更新b，那么b也不会更新，导致时间上的浪费。

```python
import collections
from collections import deque
# 采用邻接表进行图的存储
n, m = map(int, input().split())
g = [[] for i in range(n+1)]
for i in range(m):
    a, b, w = map(int, input().split())
    g[a].append((b, w))

mx = float('inf')
dis = [mx] * (n+1)
vis = [False] * (n+1)

def spfa(be, ed):
    # 采用堆优化的版本
    dis[be] = 0
    vis[be] = True #表示当前点在队列中
    que = deque()
    que.append(be)
    # 每次选取be到最短某个点
    while que:
        v = que.popleft()
        vis[v] = False
        for x, d in g[v]:
            if dis[x] > dis[v] + d:
                dis[x] = dis[v] + d
                if not vis[x]:
                    que.append(x)
                    vis[x] = True
    return dis[ed]

k = spfa(1, n)
if k == mx:
    print('impossible')
else:
    print(k)
    
    

```

- 采用vis数组的目的主要是为了解决这一种情况，因为spfa的算法思想是通过被更新的点来更新其他点，但是会出现一种情况，很多点都指向同一个点，所以会导致该点会加入队列多次。
- 所以此时需要记录一下队列中是否已经有该点了，如果有就没有必要再次加入了。

## spfa 判断负环

```python
import collections
from collections import deque
# 采用邻接表进行图的存储
n, m = map(int, input().split())
g = [[] for i in range(n+1)]
for i in range(m):
    a, b, w = map(int, input().split())
    g[a].append((b, w))
cnt = [0] * (n+1)
mx = float('inf')
dis = [0] * (n+1)
vis = [False] * (n+1)

def spfa(be, ed):
    que = deque()
    for i in range(n+1):
        vis[i] = True
        que.append(i)
    # 每次选取be到最短某个点
    while que:
        v = que.popleft()
        vis[v] = False
        for x, d in g[v]:
            if dis[x] > dis[v] + d:
                dis[x] = dis[v] + d
                cnt[x] = cnt[v] + 1
                if cnt[x] >= n:
                    return True
                if not vis[x]:
                    que.append(x)
                    vis[x] = True
    return False

if spfa(1, n):
    print('Yes')
else:
    print('No')
```

## 朴素版prim

- 和 $dijkstra$ 算法思想一样

```python
n, m = map(int, input().split())
mx = float('inf')
g = [[mx] * (n+1) for i in range(n+1)]
vis = [False] * (n+1)
dis = [mx] * (n+1)
def prim(be):
    res = 0
    dis[be] = 0
    for i in range(1,n+1):
        t = -1
        # 选取到已经加入集合中的点的最小的边
        for j in range(1, n+1):
            if not vis[j] and (t == -1 or dis[t] > dis[j]):
                t=j
        vis[t] = True
        if dis[t] == mx:
            return mx
        res += dis[t]
        # 通过这个点来更新剩下的边到集合的距离
        for k in range(n+1):
            dis[k] = min(dis[k], g[t][k])
    return res

for i in range(m):
    a, b, c = map(int, input().split())
    g[a][b] = g[b][a] = min(g[a][b], c)
k = prim(1)
if k == mx:
    print('impossible')
else:
    print(k)
```

## kruskal算法求最小生成树

```python
n, m = map(int, input().split())
parent = list(range(n+1))
# edge 数组用来存边
edge = []
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

for i in range(m):
    a, b, w = map(int, input().split())
    edge.append([a, b, w])
edge.sort(key = lambda x:x[2])
res = 0
cnt = 0
for i in range(m):
    a, b, w = edge[i]
    a, b = find(a), find(b)
    if a != b:
        parent[a] = b
        cnt += 1
        res += w
if cnt != n-1:
    print('impossible')
else:
    print(res)
```

## 拓扑排序

```python
import collections
from collections import deque
from collections import defaultdict
n, m = map(int, input().split())
g = defaultdict(list)
e = [0] * (n+1)
res = []
def topsort():
    que = deque()
    for i in range(1, n+1):
        if e[i] == 0:
            que.append(i)
    while que:
        ans = que.popleft()
        res.append(ans)
        for x in g[ans]:
            e[x] -= 1
            if e[x] == 0:
                que.append(x)
    return len(res) == n    
        
for i in range(m):
    a, b = map(int, input().split())
    g[a].append(b)
    e[b] += 1
if topsort():
    for x in res:
        print(x, end=' ')
else:
    print(-1)
```

## 二分图最大匹配(匈牙利算法)

- 主要是为了解决二分图的最大匹配问题
- 二分图的匹配：给定一个二分图 G，在 G 的一个子图 M 中，M 的边集 {E} 中的任意两条边都不依附于同一个顶点，则称 M 是一个匹配。
- 二分图的最大匹配：所有匹配中包含边数最多的一组匹配被称为二分图的最大匹配，其边数即为最大匹配数。

```python
n1, n2, m = map(int, input().split())
g = [[] for i in range(n1+1)]
match = [0] * (n2+1)
st = [False] * (n2+1)

def find(x):
    for j in g[x]:
        if not st[j]:
            st[j] = True
            if not match[j] or find(match[j]):
                match[j] = x
                return True
    return False
    
for i in range(m):
    a, b = map(int, input().split())
    g[a].append(b)
cnt = 0
for i in range(1, n1+1):
    st = [False] * (n2+1)
    if find(i):
        cnt += 1
print(cnt)
```

## 图中最短环

## 染色法判定二分图
