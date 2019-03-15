class Solution(object):
    def largestRectangleArea(self, heights):  # 84. 柱状图中最大的矩形
        """
        :type heights: List[int]
        :rtype: int
        """
        # 递增栈，存index，向后扫描，遇到高的矩形，最大面积肯定能递增，所以直接入栈
        # 遇到低的矩形，计算面积：弹出栈顶，若栈空，说明之前弹出的高度就是最小值，最小值*宽度即可
        # 若栈不空，依次计算栈内元素到当前指针处的面积，即栈内每个高度*宽度
        heights.append(0)  # 便于最后一次计算
        st = []  # 辅助 递增栈
        l = len(heights)
        point = 0
        maxnum = 0
        while point < l:
            if len(st) == 0 or heights[point] >= heights[st[-1]]: # 比栈顶的高
                st.append(point)
                point += 1
            else: # 比栈顶低
                tempindex = st[-1]
                st.pop()
                if len(st) == 0:
                    tempSum = heights[tempindex] * point
                else:
                    tempSum = heights[tempindex] * (point - st[-1] - 1)
                maxnum = max(maxnum, tempSum)
        return maxnum


if __name__ == '__main__':
    a = Solution()
    b = a.largestRectangleArea([2, 1, 5, 6, 2, 3])
    print(b)
