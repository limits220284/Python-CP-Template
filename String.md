## KMP

- $O(n)$ 时间内找到匹配的字符串

```python
n = int(input())
p = input()
m = int(input())
s = input()
ne = [-1] * n
i, j = 1, -1
#求解next数组
while i < n:
    while j != -1 and p[i] != p[j+1]:
        j = ne[j]
    if p[i] == p[j+1]:
        j += 1
    ne[i] = j
    i += 1
#进行匹配
i, j = 0, -1
while i < m:
    while j != -1 and s[i] != p[j+1]:
        j = ne[j]
    if s[i] == p[j+1]:
        j += 1
    if j == n-1:
        print(i-n+1, end = ' ')
        j = ne[j]
    i += 1
```
