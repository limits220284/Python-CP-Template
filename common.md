## 快排

- 快速排序的主要思想是每次选择一个锚点，然后确定锚点的位置
- 递归的排序左边和右边

```python
n = map(int, input())
num = list(map(int, input().split()))

def quicksort(l, r):
    if l >= r:
        return 
    x=num[(l + r) // 2]
    i, j = l-1, r+1
    while i < j:
        while True:
            i += 1
            if num[i] >= x:break
        while True:
            j -= 1
            if num[j] <= x:break
        if i < j:
            num[i], num[j] = num[j], num[i]
    quicksort(l, j)
    quicksort(j+1, r)
quicksort(0, len(num)-1)
for i in num:
  print(i, end = ' ')
```

## 快速选择排序

# 二维前缀和

```python
## m, n, q 是矩阵长 宽 和查询次数
m, n, q = map(int, input().split())
## mat是要进行求前缀和的矩阵
mat=[]
for i in range(m):
    mat.append(list(map(int,input().split())))
arr=[[0] * (n+1) for _ in range(m+1)]

#计算前缀和数组
for i in range(m):
    for j in range(n):
        arr[i+1][j+1] = arr[i][j+1] + arr[i+1][j] - arr[i][j] + mat[i][j]
for i in range(q):
    x1, y1, x2, y2 = map(int, input().split())
    print(arr[x2][y2] - arr[x1-1][y2] - arr[x2][y1-1] + arr[x1-1][y1-1])
```

## 一维差分

```python
n, m = map(int,input().split())

nums = list(map(int, input().split()))

arr = [0] * (n+2)
def insert(l, r, c):
    arr[l] += c
    arr[r+1] -= c
#构造a
for i in range(n):
    insert(i+1, i+1, nums[i])#相当于直接在该位置上进行加数
for i in range(m):
    l, r, c = map(int, input().split())
    insert(l, r, c)
#还原
for i in range(1, len(arr)):
    #原来的数组是差分数组的前缀和
    arr[i] = arr[i-1] + arr[i]
print(' '.join(str(x) for x in arr[1:-1]))
```

# 二维差分

```python
m, n, k = map(int, input().split())
nums = []
for i in range(m):
    nums.append(list(map(int,input().split())))
arr=[[0] * (n+2) for _ in range(m+2)]

def insert(x1, y1, x2, y2, c):
    arr[x1][y1] += c
    arr[x2+1][y2+1] += c
    arr[x2+1][y1] -= c
    arr[x1][y2+1] -= c

for i in range(m):
    for j in range(n):
        insert(i+1, j+1, i+1, j+1, nums[i][j])

for i in range(k):
    x1, y1, x2, y2, c = map(int, input().split())
    insert(x1, y1, x2, y2, c)
    
for i in range(1, m+1):
    for j in range(1, n+1):
        arr[i][j] = arr[i-1][j] + arr[i][j-1] - arr[i-1][j-1] + arr[i][j]

for i in range(1, m+1):
    print(' '.join(str(x) for x in arr[i][1:-1]))
```

# 归并排序

- 确定分解点
- 递归
- 将两个有序的数组合并成一个有序的数组

``` python
n = int(input())
nums = list(map(int, input().split()))
tmp = [0] * n

def merge_sort(l, r):
    if l >= r:
        return
    mid=(l + r) // 2
    merge_sort(l, mid)
    merge_sort(mid+1, r)
    k, i, j = 0, l, mid+1
    while i <= mid and j <= r:
        if nums[i] <= nums[j]:
            tmp[k] = nums[i]
            k += 1
            i += 1
        else:
            tmp[k] = nums[j]
            k += 1
            j += 1
    while i <= mid:
        tmp[k] = nums[i]
        k += 1
        i += 1
    while j <= r:
        tmp[k] = nums[j]
        k += 1
        j += 1
    nums[l:r+1] = tmp[0:k][::]

merge_sort(0, n-1)
for i in range(n):
    print(nums[i], end=' ') 
```
