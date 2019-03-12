def twoSum(self, nums, target):  # 1 两数之和
    hashDit = {}  # 通过字典实现哈希，存的是index,一般情况下查找的近似复杂度为O(1)
    for index, num in enumerate(nums):  # 总体时间复杂度O(n)
        answer = target - num
        if answer in hashDit:
            return [hashDit[answer], index]
        else:
            hashDit[num] = index


def reverse(x):  # 7 数字翻转
    flag = 1  # 是否正数
    if x < 0:
        flag = -1
        x = -x
    output = 0
    while 1:
        if x >= 1:
            output = 10 * output + x % 10
            x = x // 10
        else:
            break
    if flag == -1:
        output = -output
    if output > 2 ** 31 - 1 or output < -2 ** 31:
        return 0
    return output


def isPalindrome(x):  # 9 回文数
    tempX = x
    if x < 0:
        return False
    output = 0
    while 1:
        if x >= 1:
            output = 10 * output + x % 10
            x = x // 10
        else:
            break

    if output == tempX:
        return True
    else:
        return False
    # if x < 0: # 考虑字符串
    #     return False
    # else:
    #     y = str(x)[::-1]
    #     if y == str(x):
    #         return True
    #     else:
    #         return False


def romanToInt(s):  # 13 罗马数字转数字
    # specialDir = {"IV":4,"IX":9,"XL":40,"XC":90,"CD":400,"CM":900}
    # numDir = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
    # outputNum = 0
    # i = 0
    # while len(s) > 1:
    #     if i >= len(s)-1:
    #         break
    #     tempstr = s[i]+s[i+1]
    #     try:
    #         outputNum = outputNum + specialDir[tempstr]
    #         s = s.replace(tempstr, "")
    #         i =0
    #     except:
    #         i = i+1
    #         continue
    # for i in range(len(s)):
    #     outputNum = outputNum + numDir[s[i]]
    # return outputNum

    # 感觉比较好
    a = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    ans = 0
    for i in range(len(s)):
        if i < len(s) - 1 and a[s[i]] < a[s[i + 1]]:
            ans -= a[s[i]]
        else:
            ans += a[s[i]]
    return ans


def longestCommonPrefix(strs):  # 14 最长公共前缀
    # 扫描法，n个长为m的串复杂度为O(nm)
    commonStr = ""
    if len(strs) == 0:  # 空列表
        return commonStr
    if len(strs) == 1:  # 只有一个串
        return strs[0]
    if len(strs[0]) == 0:  # 多个串 但是第一个串是空的
        return commonStr
    for j in range(len(strs[0])):  # 第一个串从前到后的字母
        for i in range(1, len(strs)):  # 所有串
            try:
                if strs[0][j] != strs[i][j]:
                    return commonStr
                elif i == len(strs) - 1:
                    commonStr = commonStr + strs[0][j]
            except:
                return commonStr
    return commonStr

    # 最长公共前缀LCP(S1 ... Sn) = LCP(S1 ... Smid) = LCP(Smid+1 ... Sn)
    # 分治 分而治之 分治这n个串
    # 复杂度递推公式：T(n) = 2*T(n/2)+O(m) 所以时间总复杂度还是O(mn)

    # # 二分查找法：二分串，先看前半部分，不是公共的话就返回，是的话再看后半部分
    # if len(strs) == 0:  # 空列表
    #     return ""
    # if len(strs) == 1:  # 只有一个串
    #     return strs[0]
    # minlen = 2 ** 31 - 1  # 先找最短长度
    # for x in strs:
    #     if len(x) < minlen:
    #         minlen = len(x)
    # print(minlen)
    # left = 0
    # right = minlen - 1
    # commonStr = ""
    # while (left <= right): #迭代log(n)次，每次比较mn次，所以总O(logN*MN)
    #     mid = (left + right) // 2
    #     tempStr = ""
    #     for i in range(mid + 1):
    #         tempStr = tempStr + strs[0][i]
    #     for i in range(1, len(strs)):
    #         if strs[i].find(tempStr) != 0:
    #             right = mid - 1
    #             break
    #         elif i == len(strs) - 1:
    #             commonStr = tempStr
    #             left = mid + 1
    # return commonStr

    # 还可以考虑 字典树 Trie Tree 建立树的复杂度为O(MN),查询LCP时间O(M)


def isValid(s):  # 20 有效的括号
    # slen = len(s)
    # if slen == 0:
    #     return True
    # if slen % 2 == 1:
    #     return False
    # stackList = []
    # for i in range(slen):
    #     if len(stackList) == 0:
    #         if s[i] == ')' or s[i] == '}' or s[i] == ']':
    #             return False
    #     if s[i] == '(' or s[i] == '{' or s[i] == '[':
    #         stackList.append(s[i])
    #     elif (stackList[-1] == '(' and s[i] == ')') or (stackList[-1] == '{' and s[i] == '}') or (
    #             stackList[-1] == '[' and s[i] == ']'):
    #         stackList.pop()
    #     else:
    #         return False
    # if len(stackList) == 0:
    #     return True
    # else:
    #     return False

    # 官方
    stack = []
    mapping = {")": "(", "}": "{", "]": "["}
    for char in s:
        if char in mapping:
            top_element = stack.pop() if stack else '#'
            if mapping[char] != top_element:
                return False
        else:
            stack.append(char)
    return not stack


class ListNode:  # 链表
    def __init__(self, x):
        self.val = x
        self.next = None


# 链表实验
root = ListNode(2)
n1 = ListNode(4)
n2 = ListNode(5)
root.next = n1
n1.next = n2
# while root:
#     print(root.val)
#     root=root.next
root2 = ListNode(1)
m1 = ListNode(4)
m2 = ListNode(6)
root2.next = m1
m1.next = m2


# while root2:
#     print(root2.val)
#     root2=root2.next

def mergeTwoLists(l1, l2):  # 21 合并两个有序链表,l1 l2类型是ListNode
    newRoot = ListNode(-1)
    if l1 == None and l2 == None:
        return None
    newNodes = newRoot
    while (l1 is not None) or (l2 is not None):  # 不同时为空
        if l1 is None:
            newNodes.next = l2
            break
        elif l2 is None:
            newNodes.next = l1
            break
        elif l1.val <= l2.val:
            newNodes.next = l1
            l1 = l1.next
        else:
            newNodes.next = l2
            l2 = l2.next
        newNodes = newNodes.next
    return newRoot.next


# 二叉树
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def removeDuplicates(self, nums):  # 26删除数组重复项
        lastNum = 0.1
        i = 0
        while i < len(nums):
            if nums[i] == lastNum:
                nums.pop(i)
            else:
                lastNum = nums[i]
                i = i + 1
        return i

    def removeElement(self, nums, val):  # 27移除元素
        nums.sort()
        try:
            index = nums.index(val)
            num = nums.count(val)
            for i in range(num):
                nums.pop(index)
            return len(nums)
        except:
            return len(nums)

    def searchInsert(self, nums, target):  # 35 搜索插入位置
        if len(nums) == 0:
            return 0
        try:
            index = nums.index(target)
            return index
        except:

            left = 0
            right = len(nums) - 1
            while left <= right:
                mid = (left + right) // 2
                if target <= nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            return left

        # #评论区简洁做法
        # if target in nums:
        #     return nums.index(target)
        # else:
        #     nums.append(target)
        #     nums.sort()
        #     return nums.index(target)

    def strStr(self, haystack, needle):  # 28实现strStr() 子串匹配
        if len(needle) == 0:
            return 0
        else:
            index = haystack.find(needle)
            return index

    def countAndSay(self, n):  # 38 报数 递归
        if n == 1:
            return '1'
        else:
            lastStr = Solution.countAndSay(self, n - 1)
            strnum = 0  # 重复个数
            numstr = lastStr[0]  # 当前重复的元素
            outputstr = ''
            for i in range(len(lastStr) + 1):
                if i == len(lastStr):
                    outputstr = outputstr + str(strnum) + numstr
                else:
                    if lastStr[i] == numstr:
                        strnum = strnum + 1
                    else:
                        outputstr = outputstr + str(strnum) + numstr
                        numstr = lastStr[i]
                        strnum = 1
            return outputstr

    def maxSubArray(self, nums):  # 53 最大子序和
        # 有点难。。O(n)复杂度，动态规划的例子，维基：最大子数列问题
        # 定义一个函数f(n)，以第n个数为结束点的子数列的最大和，
        # 所以f(n) = max(f(n-1) + A[n], A[n]); 以f(n-1)正负分类
        # 再取fn的最大值
        tempMaxSum = nums[0]
        sumOutPut = nums[0]
        for num in nums[1:]:
            tempMaxSum = max(num, tempMaxSum + num)
            sumOutPut = max(tempMaxSum, sumOutPut)
        return sumOutPut

    def lengthOfLastWord(self, s):  # 58. 最后一个单词的长度
        if len(s) == 0:
            return 0
        else:
            sList = s.split()
            if len(sList) == 0:
                return 0
            else:
                return len(sList[-1])

    def plusOne(self, digits):  # 66. 加一
        flag = 1  # 进位
        for i in range(len(digits) - 1, -2, -1):
            if flag == 1:
                if i >= 0:
                    if digits[i] == 9:
                        digits[i] = 0
                        flag = 1
                    else:
                        digits[i] = digits[i] + 1
                        flag = 0
                        break
                else:  # 首位还需要进位
                    digits.insert(0, 1)
        return digits

    def addBinary(self, a, b):  # 67. 二进制求和
        if len(a) <= len(b):
            len1 = len(a)
            len2 = len(b)
            short = a
            long = b
        else:
            len1 = len(b)
            len2 = len(a)
            short = b
            long = a
        outstr = ''
        i = len1 - 1
        flag = 0  # 进位
        while i >= 0:
            if int(short[i]) + int(long[i + len2 - len1]) + flag >= 2:
                outstr = str((int(short[i]) + int(long[i + len2 - len1]) + flag) % 2) + outstr
                flag = 1
            else:
                outstr = str((int(short[i]) + int(long[i + len2 - len1]) + flag) % 2) + outstr
                flag = 0
            i = i - 1
        i = len2 - len1 - 1
        while i >= 0:
            if int(long[i]) + flag >= 2:
                outstr = str((int(long[i]) + flag) % 2) + outstr
                flag = 1
            else:
                outstr = str((int(long[i]) + flag) % 2) + outstr
                flag = 0
            i = i - 1
        if flag == 1:
            outstr = '1' + outstr
        return outstr

    def mySqrt(self, x):  # 69. x 的平方根
        if x == 1:
            return 1
        left = 1
        right = x - 1
        while left <= right:
            mid = (left + right) // 2
            if mid * mid > x:
                right = mid - 1
            elif mid * mid < x:
                left = mid + 1
            else:
                return mid
        return left - 1

    def climbStairs(self, n):  # 70. 爬楼梯 就是斐波那契
        if n == 1:
            return 1
        elif n == 2:
            return 2
        else:
            tempnum1 = 1
            tempnum2 = 2
            for i in range(3, n + 1):
                tempnum = tempnum2
                tempnum2 = tempnum1 + tempnum2
                tempnum1 = tempnum
            return tempnum2

    def deleteDuplicates(self, head):  # 83. 删除排序链表中的重复元素
        if head is None:
            return None
        else:
            tempnode = head
            while head.next is not None:  # 至少两个节点
                if head.next.val == head.val:
                    if head.next.next is not None:
                        head.next = head.next.next
                    else:
                        head.next = None
                        break
                else:
                    head = head.next
            return tempnode

    def merge(self, nums1, m, nums2, n):  # 88. 合并两个有序数组
        for i in range(m, m + n):
            nums1[i] = nums2[i - m]
        nums1.sort()

    def isSameTree(self, p, q):  # 100. 相同的树
        # 一边空的话肯定FALSE
        if (p is None and q is not None) or (q is None and p is not None):
            return False
        # 都不空 递归
        while (p is not None) and (q is not None):
            if p.val != q.val:
                return False
            elif self.isSameTree(p.left, q.left):
                return self.isSameTree(p.right, q.right)
            else:
                return False
        return True

    def nodeDuiChen(self, left, right):  # 101用到，判断两个节点是否镜像
        if left is None and right is None:
            return True
        if (left is None and right is not None) or (left is not None and right is None):
            return False
        if left.val != right.val:  # 都不空的话
            return False
        if self.nodeDuiChen(left.left, right.right):
            return self.nodeDuiChen(left.right, right.left)
        return False

    def isSymmetric(self, root):  # 101. 对称二叉树
        if root is None:
            return True
        if root.left is None and root.right is None:
            return True
        if (root.left is None and root.right is not None) or (root.left is not None and root.right is None):
            return False
        left = root.left  # 左半区域
        right = root.right
        if self.nodeDuiChen(left, right):
            return True
        return False

    def maxDepth(self, root):  # 104. 二叉树的最大深度
        if root is None:
            return 0
        if root.left is None and root.right is None:  # 左右都空，叶子
            return 1
        if root.left is None and root.right is not None:  # 左空右不空
            return 1 + self.maxDepth(root.right)
        if root.left is not None and root.right is None:  # 右空左不空
            return 1 + self.maxDepth(root.left)
        # 左右都不空
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))

    def levelOrderBottom(self, root):  # 107 二叉树的层次遍历
        if root is None:
            return []
        outList = [[root.val]]
        NodeList = []  # 当前层的节点列表
        if root.left is not None:
            NodeList.append(root.left)
        if root.right is not None:
            NodeList.append(root.right)
        while len(NodeList) != 0:
            tempNumList = []
            nextLevelNodeList = []  # 下一层
            for node in NodeList:
                tempNumList.append(node.val)
                if node.left is not None:
                    nextLevelNodeList.append(node.left)
                if node.right is not None:
                    nextLevelNodeList.append(node.right)
            outList.insert(0, tempNumList)  # 插入队首
            NodeList = nextLevelNodeList
        return outList

    def sortedArrayToBST(self, nums):  # 108. 将有序数组转换为二叉搜索树
        # 思路：一直二分，中间的就是要插入的节点
        lenList = len(nums)
        if lenList == 0:
            return None
        if lenList == 1:
            return TreeNode(nums[0])
        if lenList == 2:
            node1 = TreeNode(nums[1])
            node2 = TreeNode(nums[0])
            node1.left = node2
            return node1
        else:
            mid = lenList // 2
            node1 = TreeNode(nums[mid])
            node2 = self.sortedArrayToBST(nums[0:mid])
            node3 = self.sortedArrayToBST(nums[mid + 1:])
            node1.left = node2
            node1.right = node3
            return node1

    def maxDepthNode(self, root):  # 104 改. 二叉树的最大深度并且赋值,110用
        if root is None:
            return 0
        if root.left is None and root.right is None:  # 左右都空，叶子
            root.val = 1
            return 1
        if root.left is None and root.right is not None:  # 左空右不空
            root.val = 1 + self.maxDepth(root.right)
            return root.val
        if root.left is not None and root.right is None:  # 右空左不空
            root.val = 1 + self.maxDepth(root.left)
            return root.val
        # 左右都不空
        root.val = 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
        return root.val

    def isBalanced(self, root):  # 110. 平衡二叉树
        # 先递归求每个节点深度，再
        if root is None:
            return True
        if root.left is None and root.right is None:  # 左右都是空
            return True
        self.maxDepthNode(root)
        if root.left is None:  # 左空
            if root.right.val > 1:
                return False
            else:
                return self.isBalanced(root.right)
        if root.right is None:  # 右空
            if root.left.val > 1:
                return False
            else:
                return self.isBalanced(root.left)
        if abs(root.left.val - root.right.val) <= 1:  # 左右都有
            if self.isBalanced(root.left):
                return self.isBalanced(root.right)
        return False

    def minDepth(self, root):  # 111. 二叉树的最小深度
        if root is None:
            return 0
        if root.left is None and root.right is None:
            return 1
        depth = 1
        nextNodes = [root]
        while len(nextNodes) > 0:
            tempNextList = []
            for node in nextNodes:
                if node.left is None and node.right is None:
                    return depth
                if node.left is not None:
                    tempNextList.append(node.left)
                if node.right is not None:
                    tempNextList.append(node.right)
            nextNodes = tempNextList
            depth = depth + 1

    def pathSum(self, root, num, List):  # 当前节点的路径之和，num是上一层路径和，112 用,List就是路径和列表
        if root is None:
            return 0
        if root.left is None and root.right is None:  # 叶子
            List.append(root.val + num)
            return root.val + num
        if root.left is None and root.right is not None:  # 右不空
            return self.pathSum(root.right, num + root.val, List)
        if root.left is not None and root.right is None:  # 左不空
            return self.pathSum(root.left, num + root.val, List)
        # 两边都有
        self.pathSum(root.left, num + root.val, List)
        self.pathSum(root.right, num + root.val, List)

    def hasPathSum(self, root, sum):  # 112. 路径总和
        pathSumlist = []
        self.pathSum(root, 0, pathSumlist)
        # print(li)
        if sum in pathSumlist:
            return True
        else:
            return False

    def generate(self, numRows: int):  # 118 杨辉三角
        if numRows == 0:
            return []
        if numRows == 1:
            return [[1]]
        outList = [[1]]
        lastList = [1]
        for i in range(2, numRows + 1):
            templist = []
            for j in range(i):
                if j == 0 or j == i - 1:
                    templist.append(1)
                else:
                    templist.append(lastList[j - 1] + lastList[j])
            lastList = templist
            outList.append(templist)
        return outList

    def getRow(self, rowIndex: int):  # 119 杨辉三角2
        # 组合数前一项和后一项的关系
        outList = [1] * (rowIndex + 1)
        for i in range(1, rowIndex):
            outList[i] = outList[i - 1] * (rowIndex + 1 - i) // i
        return outList

    def maxProfit(self, prices):  # 121 卖股票最佳时机
        # fn是第n天卖股票获取的最大利润，fn=max[fn_1+A[n]-A[n-1] , 0] 再求所有fn最大值，动态规划
        maxLirun = 0
        fn_1 = 0
        for i in range(1, len(prices)):
            fn_1 = max(0, fn_1 + prices[i] - prices[i - 1])
            maxLirun = max(maxLirun, fn_1)
        return maxLirun

    def maxProfit(self, prices):  # 122 卖股票最佳时机2
        if len(prices) <= 1:
            return 0
        total = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:  # 当前是下降
                total = total + prices[i] - prices[i - 1]
        return total

    def isPalindrome(self, s):  # 125. 验证回文串
        if len(s) <= 1:
            return True
        i = 0
        j = len(s) - 1
        while i < j:
            if s[i].isdigit() == False and s[i].isalpha() == False:
                i = i + 1
                continue
            if s[j].isdigit() == False and s[j].isalpha() == False:
                j = j - 1
                continue
            if s[i].lower() != s[j].lower():
                return False
            else:
                i = i + 1
                j = j - 1
        return True

    def singleNumber(self, nums):  # 136 只出现一次的数字
        # nums.sort()
        # for i in range(0,len(nums)-1,2):
        #     if nums[i] != nums[i+1]:
        #         return nums[i]
        # return nums[-1]

        # 异或！相同的数会异或为0，所以最后结果就是要找的数字
        a = 0
        for num in nums:
            a = a ^ num
        return a

    def hasCycle(self, head):  # 141 环形链表
        """
        :type head: ListNode
        :rtype: bool
        """
        # 快慢指针可以判断是否包含环，如果相遇代表有环
        if head is None or head.next is None:
            return False
        fast = head.next
        slow = head
        while slow != fast:
            if fast.next is None or fast.next.next is None:
                return False
            fast = fast.next.next
            slow = slow.next
        return True

    def getIntersectionNode(self, headA, headB):  # 160 相交链表
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        lenA = 0
        lenB = 0
        temp = headA
        while temp is not None:
            lenA = lenA + 1
            if temp.next is not None:
                temp = temp.next
            else:
                break
        temp = headB
        while temp is not None:
            lenB = lenB + 1
            if temp.next is not None:
                temp = temp.next
            else:
                break
        # print(lenA,lenB)
        if lenA < lenB:
            while lenB > lenA:
                headB = headB.next
                lenB = lenB - 1
        else:
            while lenB < lenA:
                headA = headA.next
                lenA = lenA - 1
        while lenA >= 0:
            if headA == headB:
                return headA
            else:
                headA = headA.next
                headB = headB.next
                lenA = lenA - 1
        return null

    def twoSum(self, numbers, target):  # 167 两数之和2,升序序列
        i = 0
        j = len(numbers) - 1
        while i != j:
            if numbers[i] + numbers[j] > target:
                j = j - 1
            elif numbers[i] + numbers[j] < target:
                i = i + 1
            else:
                return [i + 1, j + 1]

    def convertToTitle(self, n):  # 168 excel表列名称
        dic = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L',
               13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'T', 21: 'U', 22: 'V', 23: 'W',
               24: 'X', 25: 'Y', 26: 'Z'}
        str = ''
        while n > 26:
            temp = (n - 1) // 26
            mi = 1
            while temp > 26:
                temp = (temp - 1) // 26
                mi = mi + 1
            str = str + dic[temp]
            n = n - 26 ** mi * temp
        str = str + dic[n]
        return str

    def majorityElement(self, nums):  # 169 众数
        nums.sort()
        return nums[len(nums) // 2]

    def titleToNumber(self, s):  # 171. Excel表列序号
        """
        :type s: str
        :rtype: int
        """
        num = 0
        for i in range(len(s)):
            num += (ord(s[-i - 1]) - 64) * 26 ** i
        return num

    def trailingZeroes(self, n):  # 172. 阶乘后的零
        """
        :type n: int
        :rtype: int
        """
        # 2 5 组合有0，2的个数肯定比5多，所以看质因数有几个5
        mi = 0
        while n >= 5:
            n /= 5
            mi += n
        return mi

    def rotate(self, nums, k):  # 189. 旋转数组
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        # 右移数组 可以考虑 三次翻转，先全部翻转，变成倒序
        # 再前k个翻转，正序；后n-k个翻转，正序
        L = len(nums)
        if k % L != 0:  # 否则不用翻转
            k %= L  # 移动长度大于数组长度的取模
            nums[:] = nums[::-1]
            nums[:k] = nums[k - 1::-1]
            nums[k:] = nums[L - 1:k - 1:-1]

    def reverseBits(self, n):  # 190. 颠倒二进制位
        x = str(bin(n))[2:]  # 转二进制是从第三位开始的
        x = x[::-1]
        l = len(x)
        if l < 32:  # 少于32位用0补齐
            d = 32 - l
            x = x + d * '0'
        return (int(x, 2))


class MinStack:  # 155 最小栈

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.minlist = []
        self.stack = []

    def push(self, x: int) -> None:
        self.minlist.append(x)
        self.minlist.sort()
        self.stack.append(x)
        # print(self.stack)

    def pop(self) -> None:
        self.minlist.pop(self.minlist.index(self.stack[-1]))
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.minlist[0]


if __name__ == '__main__':
    # root = ListNode(2)
    # n1 = ListNode(4)
    # n2 = ListNode(4)
    # n3 = ListNode(4)
    # root.next = n1
    # n1.next = n2
    # n2.next = n3

    r1 = TreeNode(1)
    r2 = TreeNode(2)
    r3 = TreeNode(3)
    r1.left = r2
    r1.right = r3

    t1 = TreeNode(1)
    t2 = TreeNode(2)
    t3 = TreeNode(3)
    t1.left = t2
    t1.right = t3

    # a = mergeTwoLists(root,root2)
    # a = [1,2,3,4,5,6,7,8,9]
    # a.pop([2,3,4])

    a = Solution()
    A = [1, 2, 3, 4, 5, 6, 7]
    b = a.rotate(A, 3)
    # while b:
    #     print(b.val)
    #     b = b.next
    print(A)
