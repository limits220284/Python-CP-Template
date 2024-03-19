## 单调栈(更新一下如何同时求左右两边最大的数和最小的数)

- 适用于找到某个数左边或者右边最近的`最大的数` 或者`最小的数`

```python
n = int(input())
nums = list(map(int, input().split()))
stk = [-1]
for i in range(n):
    while stk and stk[-1] >= nums[i]:
        stk.pop()
    print(stk[-1], end = ' ')
    stk.append(nums[i])
```

## 1、单调队列求区间最值

- 可以在队列中维护一个单调的数列
- 窗口内的最小值肯定出现在队头，因为在进行插入元素时，要比较当前该数字和队尾元素的大小，如果队尾元素大于该元素，需要删除掉，这样每次插入数字都会把最小的元素保留下来，即使当前插入的元素不是最小的，也能保证最小的元素在队列中

```python
import collections
from collections import deque
n, k = map(int, input().split())
nums = list(map(int, input().split()))
que = deque()

for i in range(n):
    if que and que[0] < i-k+1: que.popleft()
    while que and nums[que[-1]] >= nums[i]:
        que.pop()
    que.append(i)
    if i >= k-1:
        print(nums[que[0]], end = ' ')
print()
que = deque()
for i in range(n):
    if que and que[0] < i-k+1: que.popleft()
    while que and nums[que[-1]] <= nums[i]:
        que.pop()
    que.append(i)
    if i >= k-1:
        print(nums[que[0]], end = ' ')
```

## 字典树

```python
Trie = {}
def insert(s):
    Node = Trie
    for x in s:
        Node = Node.setdefault(x, {})#如果存在ch则返回原本的值，否则返回{}
    Node['count'] = Node.get('count', 0) + 1

def query(s):
    Node = Trie
    for x in s:
        if x not in Node:
            return 0
        Node = Node[x]
    return Node.get('count', 0)#
    
n = int(input())
for i in range(n):
    a, b = input().split()
    if a == "I":
        insert(b)
    else:
        print(query(b))

```

## 并查集

```python
# 并查集模板
class DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.size = [1] * n
        self.cnt = n
    def find(self, x):
        if x != self.p[x]:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self, x, y):
        x, y = self.find(x), self.find(y)
        if x == y: return
        if self.size[x] < self.size[y]:
            x, y = y, x
        self.size[x] += self.size[y]
        self.p[y] = x
        self.cnt -= 1
```

## 字符串哈希

## 维护前缀和

```python
# 从1开始
class FenwickTree:
    def __init__(self, n):
        self.n = n
        self.tr = [0] * (n+1)
    def lowbit(self, x):
        return x & -x
    
    def add(self, x, c):
        i = x
        while i <= self.n:
            self.tr[i] += c
            i += self.lowbit(i)
    def sum(self, x):
        res = 0
        i = x
        while i > 0:
            res += self.tr[i]
            i -= self.lowbit(i)
        return res
    def query(self, l, r):
        return self.sum(r) - self.sum(l-1)

```

## 树状数组维护前缀最大值

```python

class BIT:
    def __init__(self, n: int):
        self.tree = [-inf] * n

    def update(self, i: int, val: int) -> None:
        while i < len(self.tree):
            self.tree[i] = max(self.tree[i], val)
            i += i & -i

    def pre_max(self, i: int) -> int:
        mx = -inf
        while i > 0:
            mx = max(mx, self.tree[i])
            i -= i & -i
        return mx
```

## 树状数组维护后缀最大值

```c++
class BIT {
// 注意树状数组维护区间后缀最大值的写法
public:
    vector<int>tree;
    int n;
    static int lowbit(int x){
        return x & (-x);
    }
    BIT(int _n){
        n = _n;
        tree.assign(_n+1, -1);
    }
    int query(int x){
        int ans = -1;
        while(x <= n){
            ans = max(ans, tree[x]);
            x += lowbit(x);
        }
        return ans;
    }
    void update(int x, int val){
        while(x){
            tree[x] = max(tree[x], val);
            x -= lowbit(x);
        }
    }
};
```

## 普通线段树 之 维护区间最大值

- 单点修改，区间查询
- python版本

```python
class SegmentTree:
    def __init__(self, n):
        self.f = [[0, 0, -inf] for i in range(4 * n)]
    
    def pushup(self, u):
        self.f[u][2] = max(self.f[u * 2][2], self.f[u * 2 + 1][2])
        
    def buildTree(self, u, l, r):
        self.f[u][0], self.f[u][1] = l, r
        if l >= r: 
            return 
        mid = (self.f[u][0] + self.f[u][1]) // 2
        self.buildTree(2 * u, l, mid)
        self.buildTree(2 * u + 1, mid + 1, r) 
        self.pushup(u)
    
    def update(self, u, x, v):
        if self.f[u][0] == x and self.f[u][1] == x:
            self.f[u][2] = v
            return 
        mid = (self.f[u][0] + self.f[u][1]) // 2
        if x <= mid: 
            self.update(2 * u, x, v)
        elif x > mid:
            self.update(2 * u + 1, x, v)
        self.pushup(u)
    
    def query(self, u, l, r):
        if self.f[u][0] >= l and self.f[u][1] <= r:
            return self.f[u][2]
        mid = (self.f[u][0] + self.f[u][1]) // 2
        v = -inf
        if l <= mid: 
            v = self.query(2 * u, l, min(r, mid))
        if mid < r: 
            v = max(v, self.query(2 * u + 1, max(mid + 1, l), r))
        return v
```

## 普通线段树 之 维护区间和

## 懒标记线段树 之 维护区间最大值

- python版本

```python
class Node:
    def __init__(self, l = 0, r = 0, mx = 0):
        self.l = l
        self.r = r
        self.mx = mx
class SegmentTree:
    def __init__(self, n):
        self.f = [Node() for i in range(4 * n)]
    
    def pushup(self, u):
        self.f[u].mx = max(self.f[u * 2].mx, self.f[u * 2 + 1].mx)
    
    def pushdown(self, u):
        if self.f[u].mx:
            self.f[u * 2].mx = max(self.f[u * 2].mx, self.f[u].mx)
            self.f[u * 2 + 1].mx = max(self.f[u * 2 + 1].mx, self.f[u].mx)
            self.f[u].mx = 0
    def build(self, u, l, r):
        self.f[u].l, self.f[u].r = l, r
        if l >= r: 
            return 
        mid = (self.f[u].l + self.f[u].r) // 2
        self.build(2 * u, l, mid)
        self.build(2 * u + 1, mid + 1, r) 
        self.pushup(u)
    
    def modify(self, u, l, r, d):
        if self.f[u].l >= l and self.f[u].r <= r:
            self.f[u].mx = max(self.f[u].mx, d)
            return
        self.pushdown(u)
        mid = (self.f[u].l + self.f[u].r) // 2
        if l <= mid:
            self.modify(u * 2, l, r, d)
        if r > mid:
            self.modify(u * 2 + 1, l, r, d)
        # self.pushup(u)
    
    def query(self, u, l, r):
        if self.f[u].l >= l and self.f[u].r <= r:
            return self.f[u].mx
        self.pushdown(u)
        mid = (self.f[u].l + self.f[u].r) // 2
        v = -inf
        if l <= mid: 
            v = self.query(2 * u, l, min(r, mid))
        if mid < r: 
            v = max(v, self.query(2 * u + 1, max(mid + 1, l), r))
        return v
```

## 懒标记线段树 之 维护区间最小值

## 懒标记线段树 之 维护区间和

- 信息:
  - sum: 当前区间的总和
  - add: 懒标记，给以当前节点为根的子树中的每一个节点加上add
- python版本

```python
class Node:
    def __init__(self, l = 0, r = 0, sum = 0, add = 0):
        self.l = l
        self.r = r
        self.sum = sum
        self.add = add
class Segtree:
    def __init__(self, n):
        self.tr = [Node() for _ in range(4 * n)]
    def pushup(self, u):
        self.tr[u].sum = self.tr[u << 1].sum + self.tr[u << 1 | 1].sum
    
    def pushdown(self, u):
        root, left, right = self.tr[u], self.tr[u << 1], self.tr[u << 1 | 1]
        if root.add:
            left.add += root.add
            left.sum += (left.r - left.l + 1) * root.add
            right.add += root.add
            right.sum += (right.r - right.l + 1) * root.add
            root.add = 0
    
    def build(self, u, l, r):
        if l == r:
            self.tr[u] = Node(l, r, w[r], 0)
        else:
            self.tr[u] = Node(l, r)
            mid = (l + r) // 2
            self.build(u << 1, l, mid)
            self.build(u << 1 | 1, mid + 1, r)
            self.pushup(u)
    
    def modify(self, u, l, r, d):
        if self.tr[u].l >= l and self.tr[u].r <= r:
            self.tr[u].sum += (self.tr[u].r - self.tr[u].l + 1) * d
            self.tr[u].add += d
        else:
            self.pushdown(u)
            mid = (self.tr[u].l + self.tr[u].r) // 2
            if l <= mid:
                self.modify(u << 1, l, r, d)
            if r > mid:
                self.modify(u << 1 | 1, l, r, d)
            self.pushup(u)
    
    def query(self, u, l, r):
        ans = 0
        if self.tr[u].l >= l and self.tr[u].r <= r:
            return self.tr[u].sum
        self.pushdown(u)
        mid = (self.tr[u].l + self.tr[u].r) // 2
        if l <= mid:
            ans += self.query(u << 1, l, r)
        if r > mid:
            ans += self.query(u << 1 | 1, l, r)
        return ans
```
