# 获取列表的第二个元素
def takeSecond(elem):
    return elem[1]

# 题目：https://www.nowcoder.com/test/question/f652bf7904bf4905804fa3bc347fdd2a?pid=8537279&tid=22153875
# 说运行超过内存限制，应该已经是极限了，通过80%
if __name__ == '__main__':
    l = int(input())
    xList = []
    x = 0
    y = 0
    for i in range(l):
        x, y = map(int, input().split())
        xList.append([x,y])
    xList.sort(key=takeSecond,reverse=True)#按y轴降序
    # print(xList)
    maxX = 0
    for i in range(l):
        if xList[i][0] >= maxX:
            maxX = xList[i][0]
            print(xList[i][0],xList[i][1])