class Solution(object):
    # 回溯法 39、40 用
    def back(self, output, temp, numList, remain, begin):
        '''

        :param output: 最终输出的list
        :param temp: 每个组合的list
        :param numList: 原来的数组
        :param remain: 待搜索的需要求和的剩余值
        :param begin: 搜索起始位置
        :return: 是否有解
        '''
        if remain < 0:  # 无解
            return False
        elif remain == 0:  # 有解，已结束
            output.append(list(temp))
            return False
        else:
            for i in range(begin, len(numList)):
                temp.append(numList[i])
                flag = self.back(output, temp, numList, remain - numList[i], i)  # 最后参数：39题用i，因为元素可以无限制重复，40题用i+1，不能重复
                temp.pop()
                if flag == False:
                    break
            return True

    # 39. 组合总和 数字可重复
    def combinationSum(self, candidates, target):
        output = []
        temp = []
        candidates.sort()
        self.back(output, temp, candidates, target, 0)
        return output

    # 回溯 40
    def back2(self, output, temp, numList, remain, begin):
        if remain < 0:  # 无解
            return False
        elif remain == 0:  # 有解，已结束
            if temp not in output:
                output.append(list(temp))
            return False
        else:
            for i in range(begin, len(numList)):
                temp.append(numList[i])
                flag = self.back2(output, temp, numList, remain - numList[i], i + 1)
                temp.pop()
                if flag == False:
                    break
            return True

    # 40 组合总和 II
    def combinationSum2(self, candidates, target):
        output = []
        temp = []
        candidates.sort()
        self.back2(output, temp, candidates, target, 0)
        return output

    def CC(self, N, M):  # 组合数 C(n,m) m>=n
        tempnum = 1.0
        for i in range(1, M - N + 1):
            tempnum *= N + i
            tempnum /= i
        return int(tempnum)

    # 62. 不同路径
    def uniquePaths(self, m, n):
        # 计算组合数，总共需要走m+n-2步，m-1步横着走，n-1步竖着走，C(m-1,m+n-2)
        return self.CC(m - 1, m + n - 2)

    # 63. 不同路径 II
    def uniquePathsWithObstacles(self, obstacleGrid):

        m = len(obstacleGrid)
        if m == 0:
            return 0
        n = len(obstacleGrid[0])
        if n == 0:
            return 0

        if obstacleGrid[m - 1][n - 1] == 1:
            return 0
        # # 递归费时
        # if m == 1 and n == 1:
        #     return 1
        # if m == 1:
        #     return self.uniquePathsWithObstacles([obstacleGrid[0][:-1]])
        # if n == 1:
        #     return self.uniquePathsWithObstacles(obstacleGrid[:-1])
        # return self.uniquePathsWithObstacles([x[:-1] for x in obstacleGrid]) + self.uniquePathsWithObstacles(
        #     obstacleGrid[:-1])

        num = [[0 for i in range(n)] for j in range(m)]
        # print(num)
        for j in range(n):  # 第一行
            if obstacleGrid[0][j] == 0:
                num[0][j] = 1
            else:
                break
        for i in range(m):  # 第一列
            if obstacleGrid[i][0] == 0:
                num[i][0] = 1
            else:
                break
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j - 1] == 1 and obstacleGrid[i - 1][j] == 1:
                    num[i][j] = 0
                elif obstacleGrid[i - 1][j] == 1:  # 上面有障碍
                    num[i][j] = num[i][j - 1]
                elif obstacleGrid[i][j - 1] == 1:  # 左面有障碍
                    num[i][j] = num[i - 1][j]
                else:
                    num[i][j] = num[i - 1][j] + num[i][j - 1]
        return num[m - 1][n - 1]


if __name__ == "__main__":
    a = Solution()
    A = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    A[2][2] = 0
    B = []
    b = a.uniquePathsWithObstacles([[0, 0], [1, 1], [0, 0]])
    print(b)
