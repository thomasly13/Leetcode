

def test(arg1, arg2):
    return arg1 + arg2 

print(test(1, 2))


def concat(str1, str2):
    return (f'I am a {str1}, but also a {str2}')

print(concat("dolphin", "gorilla"))


def loop(array):
    for w in array:
        print(w)


array = ["Aligator", "Bannana", "Cat", "Dolphin"]

loop(array)


# Merge alternatively
    def mergeAlternately(self, word1, word2):
        final_word = ""

        word1_len = len(word1)
        word2_len = len(word2)

        counter1 = 0

        counter2 = 0

        while counter1 < word1_len and counter2 < word2_len:
            final_word += word1[counter1] + word2[counter2]
            counter1 += 1
            counter2 += 1

        if counter1 == word1_len:
            final_word += word2[counter2:]
        else: 
            final_word += word1[counter1:]
        
        return final_word

#Can Place FLowers


    def canPlaceFlowers(self, flowerbed, n):
        
        current_bed = 0 
        flowerbed_length = len(flowerbed)

        counter = 0

        if flowerbed_length == 1 and flowerbed[0] == 0:
            counter += 1
            return (counter >= n)

        while current_bed < flowerbed_length:
            if flowerbed[current_bed] != 1:
                if current_bed == 0 and flowerbed[current_bed + 1] != 1:
                    flowerbed[current_bed] = 1
                    counter += 1
                elif flowerbed[current_bed - 1] != 1:
                    if current_bed == flowerbed_length - 1 or flowerbed[current_bed + 1] != 1:
                        flowerbed[current_bed] = 1
                        counter += 1
                current_bed += 1
            else:
                current_bed += 1
                
        return (counter >= n)

# Isomorphic Strings
    def isIsomorphic(self, s: str, t: str) -> bool:
        return len(set(s))==len(set(zip(s,t)))==len(set(t))


# Word Search
    def exist(self, board, word):
        def backtrack(i, j, k):
            if k == len(word):
                return True
            if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] != word[k]:
                return False
            
            temp = board[i][j]
            board[i][j] = ''
            
            if backtrack(i+1, j, k+1) or backtrack(i-1, j, k+1) or backtrack(i, j+1, k+1) or backtrack(i, j-1, k+1):
                return True
            
            board[i][j] = temp
            return False
        
        for i in range(len(board)):
            for j in range(len(board[0])):
                if backtrack(i, j, 0):
                    return True
        return False


# Average Wait Time
    def averageWaitingTime(self, customers: List[List[int]]) -> float:
        currentTime = 0
        totalwaitTime = 0
        
        for customer in customers:
            arrival, time = customer
            
            if currentTime < arrival:
                currentTime = arrival
                
            waitTime = currentTime + time - arrival
            totalwaitTime += waitTime
            
            currentTime += time
        
        return totalwaitTime / len(customers)

# Crawler Log Folder
    def minOperations(self, logs):
        step = 0
        for log in logs:
            if log == '../':
                if step > 0:
                    step -= 1
            elif log != './':
                step += 1
        return step

# Reverse substrings between each pair of parenthesis 
    def reverseParentheses(self, s):
        """
        :type s: str
        :rtype: str
        """
        stack = []
        
        for char in s:
            if char == ')':
                # Pop from the stack until encountering '('
                temp = []
                while stack and stack[-1] != '(':
                    temp.append(stack.pop())
                stack.pop()  # Remove the '(' from the stack
                # Reverse the characters and push them back onto the stack
                stack.extend(temp)
            else:
                # Push the character onto the stack
                stack.append(char)
        
        # Join the stack to form the final result
        return ''.join(stack)

# Directions from a Binary Tree Node to Another
def getDirections(self, root: Optional[TreeNode], startValue: int, destValue: int) -> str:
    graph = collections.defaultdict(list)
    # key: [(node,direction),()]
    
    startNode = None
    def traverse(root,parent):
        nonlocal startNode
        if not root:
            return
        
        if root.val == startValue:
            startNode = root
            
        graph[root].append((parent,"U"))
        if root.left:    
            graph[root].append((root.left,"L"))
            traverse(root.left,root)
        if root.right:
            graph[root].append((root.right,"R"))
            traverse(root.right,root)
        
    traverse(root,None)
    

# Delete Nodes and Return Forest
def delNodes(self, root: Optional[TreeNode], to_delete: List[int]) -> List[TreeNode]:

    result = []

    def dfs(parent: Optional[TreeNode], node: Optional[TreeNode]):

        nonlocal result

        if not node:

            return

        if node.val in to_delete:

            if parent:

                if parent.left == node:

                    parent.left = None

                else:

                    parent.right = None

            dfs(None, node.left)

            dfs(None, node.right)

        else:

            if not parent:

                result.append(node)

            dfs(node, node.left)

            dfs(node, node.right)

    dfs(None, root)

    return result

# Number of good leaf nodes
    def countPairs(self, root: TreeNode, distance: int) -> int:
        self.totalPairs = 0

        def dfs(tree):
            # leaf node
            if not tree.left and not tree.right:
                return [1]
            
            possible = []
            if tree.left:
                left = dfs(tree.left)
                possible+=left
            if tree.right:
                right = dfs(tree.right)
                for length in possible:
                    for length2 in right:
                        if length+length2 <= distance:
                            self.totalPairs+=1
                possible+=right
            
            return [1+length for length in possible]

        dfs(root)
        return self.totalPairs


#Sort array by frequencies
    def frequencySort(self, nums):
        freq = Counter(nums)
        return sorted(nums, key=lambda x : (freq[x], -x))


# Sort array using quick sort
    def sortArray(self, N: List[int]) -> List[int]:
        def quicksort(A, I, J):
            if J - I <= 1: return
            p = partition(A, I, J)
            quicksort(A, I, p), quicksort(A, p + 1, J)
        
        def partition(A, I, J):
            A[J-1], A[(I + J - 1)//2], i = A[(I + J - 1)//2], A[J-1], I
            for j in range(I,J):
                if A[j] < A[J-1]: A[i], A[j], i = A[j], A[i], i + 1
            A[J-1], A[i] = A[i], A[J-1]
            return i
        
        quicksort(N,0,len(N))
        return N


# Count and Say
def countAndSay(self, n):
        s = '1'
        for _ in range(n-1):
            let, temp, count = s[0], '', 0
            for l in s:
                if let == l:
                    count += 1
                else:
                    temp += str(count)+let
                    let = l
                    count = 1
            temp += str(count)+let
            s = temp
        return s


# Filling Bookshelves
    def minHeightShelves(self, books, shelfWidth):
        n = len(books)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: no books require 0 height
        
        for i in range(1, n + 1):
            total_width = 0
            max_height = 0
            for j in range(i, 0, -1):
                total_width += books[j-1][0]
                if total_width > shelfWidth:
                    break
                max_height = max(max_height, books[j-1][1])
                dp[i] = min(dp[i], dp[j-1] + max_height)
        
        return dp[n]

# Minimum Swaps to Group all 1's Together
    def minSwaps(self, nums: List[int]) -> int:
        k = nums.count(1)
        mx = cnt = sum(nums[:k])
        n = len(nums)
        for i in range(k, n + k):
            cnt += nums[i % n]
            cnt -= nums[(i - k + n) % n]
            mx = max(mx, cnt)
        return k - mx


# Range Sum of Sorted Subarray Sums
def rangeSum(self, nums: List[int], n: int, left: int, right: int) -> int:
    l = []
    for i in range(len(nums)):
        cum = 0
        for j in range(i,len(nums)):
            cum+=nums[j]
            l.append(cum)
    l.sort()
    return sum(l[left-1:right])%(10**9+7)

# Single Number 3
    def singleNumber(self, a: List[int]) -> List[int]:
        # total xor-sum
        x = reduce(xor, a)
        # first different bit for two numbers
        d = -x&x
        # one number
        y = reduce(xor, (v for v in a if d&v))
        # another number will be x^y
        return x^y, y


# Number to Words
   def numberToWords(self, num):
        less_than_twenty = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
        ten_places = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]

        if num == 0:
            return "Zero"

        def two_digit(num):
            if num < 20:
                return less_than_twenty[num]
            else:
                tens = num // 10
                ones = num % 10
                return ten_places[tens] + ('' if ones == 0 else ' ' + less_than_twenty[ones])

        def three_digit(num):
            if not num: return ''
            if not num//100: return two_digit(num)
            return less_than_twenty[num//100] + ' ' +'Hundred' + (' ' + two_digit(num%100) if num%100 else '')

        billion = num // 1000000000
        million = (num // 1000000) % 1000
        thousand = (num // 1000) % 1000
        hundred = num % 1000

        res = ''
        if billion:
            res += three_digit(billion) + ' Billion'
        
        if million:
            if res:
                res += ' '
            res += three_digit(million) + ' Million'
        
        if thousand:
            if res:
                res += ' '
            res += three_digit(thousand) + ' Thousand'
        
        if hundred:
            if res:
                res += ' '
            res += three_digit(hundred)

        return res.strip()



# Sprial Matrix 3
    def spiralMatrixIII(self, rows, cols, rStart, cStart):
        i,j = rStart, cStart

        diri, dirj = 0, 1 # directions to move
        twice = 2
        res = []
        moves = 1
        next_moves = 2

        while len(res) < (rows * cols):
            if (-1 < i < rows) and ( -1 < j < cols):
                res.append([i,j])
            
            i += diri
            j += dirj
            moves -= 1
            if moves == 0:
                diri, dirj = dirj, -diri # 90 deg Clockwise
                twice -= 1
                if twice == 0:
                    twice = 2
                    moves = next_moves
                    next_moves += 1
                else:
                    moves = next_moves - 1
        return res


# Magic Squares
    def numMagicSquaresInside(self, grid: List[List[int]]) -> int:
        n = len(grid)
        m = len(grid[0])
        if n < 3 or m < 3:
            return 0
        cnt = 0
        magic_sq = [
            [[4, 9, 2], [3, 5, 7], [8, 1, 6]],
            [[2, 7, 6], [9, 5, 1], [4, 3, 8]],
            [[6, 1, 8], [7, 5, 3], [2, 9, 4]],
            [[8, 3, 4], [1, 5, 9], [6, 7, 2]],
            [[4, 3, 8], [9, 5, 1], [2, 7, 6]],
            [[2, 9, 4], [7, 5, 3], [6, 1, 8]],
            [[6, 7, 2], [1, 5, 9], [8, 3, 4]],
            [[8, 1, 6], [3, 5, 7], [4, 9, 2]]
        ]
        for r_start in range(n - 2):
            for c_start in range(m - 2):
                subgrid = [grid[r_start + i][c_start:c_start + 3] for i in range(3)]
                if subgrid in magic_sq:
                    cnt += 1
        return cnt


# Smallest pair distance
   def smallestDistancePair(self, nums, k):
        nums.sort()
        n = len(nums)
        low, high = 0, nums[-1] - nums[0]

        def count_pairs(max_distance):
            count, j = 0, 0
            for i in range(n):
                while j < n and nums[j] - nums[i] <= max_distance:
                    j += 1
                count += j - i - 1
            return count

        while low < high:
            mid = (low + high) // 2
            if count_pairs(mid) < k:
                low = mid + 1
            else:
                high = mid

        return low


# 2 keys keyboard
    def minSteps(self, n: int) -> int:
        steps = 0
        i = 2
        while i <= n:
            while n % i == 0:
                steps += i
                n //= i
            i += 1
        return 
        

# stone game 2
    def stoneGameII(self, piles: List[int]) -> int:
        n = len(piles)
        
        dp = [[0] * (n + 1) for _ in range(n)]
        suffix_sum = [0] * n
        suffix_sum[-1] = piles[-1]
        
        for i in range(n - 2, -1, -1):
            suffix_sum[i] = suffix_sum[i + 1] + piles[i]
        
        for i in range(n - 1, -1, -1):
            for m in range(1, n + 1):
                if i + 2 * m >= n:
                    dp[i][m] = suffix_sum[i]
                else:
                    for x in range(1, 2 * m + 1):
                        dp[i][m] = max(dp[i][m], suffix_sum[i] - dp[i + x][max(m, x)])
        
        return dp[0][1]


# Nearest Palindrome
    def nearestPalindromic(self, n):
        """
        :type n: str
        :rtype: str
        """
        def palindrome(a): # to use in the 'else' below
             b = int(a + a[::-1]) if len(n) % 2 == 0 else int(a + a[-2::-1])  
             if str(b) == n: b = 0 # to check if palindrm is same as n
             return b
             
        if len(n) == 1: return str(int(n)-1) 
        elif all(i == '9' for i in n): return str(int(n)+2)
        elif int(n) > 10**(len(n)-1) -1 and int(n) <= 10**(len(n)-1)+1: return str(10**(len(n)-1) -1)
        
        else:
            x = n[:-(len(n)//2)]
            p1 = palindrome(x)
            y = str(int(x)+1)
            p2 = palindrome(y)
            y = str(int(x)-1)
            p3 = palindrome(y)

            #except when they're equal to zero, assume p2 > p1 > p3

            if abs(int(n)-p2) >= abs(int(n) - p1):
                if abs(int(n) - p1)  >= abs(int(n) - p3): return str(p3)
                else: return str(p1)
            else:
                if abs(int(n) - p2)  >= abs(int(n) - p3): return str(p3)
                else: return str(p2)



# Most stones removed with same row or column
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)

        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

class Solution:
    def removeStones(self, stones):
        n = len(stones)
        uf = UnionFind(n)

        rowMap = {}
        colMap = {}

        for i, (row, col) in enumerate(stones):
            if row in rowMap:
                uf.union(i, rowMap[row])
            else:
                rowMap[row] = i

            if col in colMap:
                uf.union(i, colMap[col])
            else:
                colMap[col] = i

        uniqueComponents = {uf.find(i) for i in range(n)}

        return n - len(uniqueComponents)

    # Sum of digits of string after conversion
        def getLucky(self, s, k):
        # Convert each character in the string to its corresponding numeric value
        number = ''
        for x in s:
            number += str(ord(x) - ord('a') + 1)
        
        # Perform the transformation `k` times
        while k > 0:
            temp = 0
            for x in number:
                temp += int(x)  # Sum the digits of the current number
            number = str(temp)  # Convert the sum back to a string
            k -= 1
        return int(number)  # Return the final result as an integer

        # Find Missing Obervations
            def missingRolls(self, rolls, mean, n):
        m = len(rolls)
        total_sum = mean * (n + m)
        observed_sum = sum(rolls)
        
        missing_sum = total_sum - observed_sum
        
        if missing_sum < n or missing_sum > 6 * n:
            return []
        
        base = missing_sum // n
        remainder = missing_sum % n
        
        result = [base] * n
        for i in range(remainder):
            result[i] += 1
        
        return result


    # Delete Nodes of Linked list in array
    class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    class Solution:
        def modifiedList(self, nums, head):
            num_set = set(nums)
            dummy = ListNode(-1)
            node = dummy
            
            while head:
                if head.val not in num_set:
                    node.next = head
                    node = node.next
                head = head.next
            node.next = None
            return dummy.next

    # Minimum bit flips to convert number
        def minBitFlips(self, start: int, goal: int) -> int:
        xor = start ^ goal
        return bin(xor).count('1')


    # XOR queries of a subarray
    def xorQueries(self, arr: List[int], queries: List[List[int]]) -> List[int]:
        n = len(arr)
        prefixXor = [0] * (n + 1)
        
        # Compute the prefix XOR array
        for i in range(n):
            prefixXor[i + 1] = prefixXor[i] ^ arr[i]
        
        result = []
        
        # Process each query
        for left, right in queries:
            result.append(prefixXor[right + 1] ^ prefixXor[left])
        
        return result

    # Longest Sub Array with amximum Bitsize AND   
        def longestSubarray(self, nums: List[int]) -> int:
        maxBitwiseAnd = max(nums)
        maxi = 1
        count = 0
        i = 0
        
        while i < len(nums):
            if nums[i] == maxBitwiseAnd:
                while i < len(nums) and nums[i] == maxBitwiseAnd:
                    count += 1
                    i += 1
                maxi = max(maxi, count)
                count = 0
            else:
                i += 1
        
        return maxi

    # Different ways to add parenthesis
        def __init__(self):
        self.memo = {}

    def diffWaysToCompute(self, expression: str) -> list:
        if expression in self.memo:
            return self.memo[expression]

        result = []
        
        for i, c in enumerate(expression):
            if c in "+-*":
                leftResults = self.diffWaysToCompute(expression[:i])
                rightResults = self.diffWaysToCompute(expression[i + 1:])

                for left in leftResults:
                    for right in rightResults:
                        if c == '+':
                            result.append(left + right)
                        elif c == '-':
                            result.append(left - right)
                        elif c == '*':
                            result.append(left * right)

        if not result:
            result.append(int(expression))

        self.memo[expression] = result
        return result

        # Shortest Palindrome 
            def shortestPalindrome(self, s: str) -> str:
        count = self.kmp(s[::-1], s)
        return s[count:][::-1] + s
    def kmp(self, txt: str, patt: str) -> int:
        new_string = patt + '#' + txt
        pi = [0] * len(new_string)
        i = 1
        k = 0
        while i < len(new_string):
            if new_string[i] == new_string[k]:
                k += 1
                pi[i] = k
                i += 1
            else:
                if k > 0:
                    k = pi[k - 1]
                else:
                    pi[i] = 0
                    i += 1
        return pi[-1]