

# 1.字符串分割

```python
class Solution:
    def countSegments(self, s: str) -> int:
        # length = len(s)
        # num =0
        # i=0
        # while i <length:
        #     if s[i]!=' ':
        #         i = i+1
        #         continue
        #     num = num+1
        #     while i<length and s[i+1] == ' ':
        #         i = i+1
        #     i = i+1
        # return num
        words = s.split(' ')
        size = len(words)
        num = size
        for i in range(0,size):
            if len(words[i]) == 0:
                num = num -1
        return num
```

# 2. GoogleNet网络结构

![2021071501](..\images\202107\2021071501.jpg)
