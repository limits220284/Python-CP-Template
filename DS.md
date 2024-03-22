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

## AVL平衡树

```python
import random
class Node:
    def __init__(self):
        self.l = self.r = 0
        self.key = self.val = 0
        self.cnt = self.size = 0    #这里初始值设为1会出错

class SortedList:
    def __init__(self, n):
        self.tr = [Node() for i in range(n)]
        self.root = self.idx = 0

    def pushup(self, p):
        self.tr[p].size = self.tr[self.tr[p].l].size + self.tr[self.tr[p].r].size + self.tr[p].cnt

    def get_node(self, key):
        self.idx += 1
        self.tr[self.idx].key, self.tr[self.idx].val, self.tr[self.idx].cnt, self.tr[self.idx].size = key, random.random(), 1, 1
        return self.idx

    def zig(self, p):  #右旋   这里的p相当于指针，每次更改节点位置都要相应更改指针位置
        q = self.tr[p].l
        self.tr[p].l = self.tr[q].r; self.tr[q].r = p; p = q
        self.pushup(st.tr[p].r); self.pushup(p)
        return p

    def zag(self, p):   #左旋
        q = self.tr[p].r
        self.tr[p].r = self.tr[q].l; self.tr[q].l = p; p = q
        self.pushup(self.tr[p].l); self.pushup(p)
        return p

    def build(self):
        self.root = self.get_node(-float("inf"))    #建立两个哨兵节点root为1,root.r为2
        self.tr[self.root].r = self.get_node(float("inf"))
        self.pushup(self.root)
        if self.tr[1].val < self.tr[2].val: root = self.zag(self.root)

    def insert(self, p, key):
        if not p: p = self.get_node(key)
        elif self.tr[p].key == key: self.tr[p].cnt += 1
        elif self.tr[p].key > key:
            self.tr[p].l = self.insert(self.tr[p].l, key)
            if self.tr[self.tr[p].l].val > self.tr[p].val: p = self.zig(p)
        else:
            self.tr[p].r = self.insert(self.tr[p].r, key)
            if self.tr[self.tr[p].r].val > self.tr[p].val: p = self.zag(p)
        self.pushup(p)
        return p

    def remove(self, p, key):
        if not p: return p
        if self.tr[p].key == key:
            if self.tr[p].cnt > 1: self.tr[p].cnt -= 1
            elif self.tr[p].l or self.tr[p].r:
                if not self.tr[p].r or self.tr[self.tr[p].l].val > self.tr[p].val:
                    p = self.zig(p)
                    self.tr[p].r = self.remove(self.tr[p].r, key)   #右旋后当前点位于原点的右子树
                else:
                    p = self.zag(p)
                    self.tr[p].l = self.remove(self.tr[p].l, key)
            else: p = 0
        elif self.tr[p].key > key: self.tr[p].l = self.remove(self.tr[p].l, key)
        else: self.tr[p].r = self.remove(self.tr[p].r, key)
        self.pushup(p)
        return p

    def get_rank_by_key(self, p, key):
        if not p: return 0   #本题中不会出现这种情况
        if self.tr[p].key == key: return self.tr[self.tr[p].l].size + 1
        if self.tr[p].key > key: return self.get_rank_by_key(self.tr[p].l, key)
        return self.tr[self.tr[p].l].size + self.tr[p].cnt + self.get_rank_by_key(self.tr[p].r, key)

    def get_key_by_rank(self, p, rank):
        if not p: return float("inf")
        if self.tr[self.tr[p].l].size >= rank: return self.get_key_by_rank(self.tr[p].l, rank)
        if self.tr[self.tr[p].l].size + self.tr[p].cnt >= rank: return self.tr[p].key
        return self.get_key_by_rank(self.tr[p].r, rank - self.tr[self.tr[p].l].size - self.tr[p].cnt)

    def get_prev(self, p, key):
        if not p: return -float("inf")
        if self.tr[p].key >= key: return self.get_prev(self.tr[p].l, key)
        return max(self.tr[p].key, self.get_prev(self.tr[p].r, key))

    def get_next(self, p, key):
        if not p: return float("inf")
        if self.tr[p].key <= key: return self.get_next(self.tr[p].r, key)
        return min(self.tr[p].key, self.get_next(self.tr[p].l, key))
```
