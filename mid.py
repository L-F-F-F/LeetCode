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


if __name__ == "__main__":
    a = Solution()
    b = a.combinationSum2([2, 3, 1], 3)
    print(b)
