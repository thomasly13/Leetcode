

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

    # Sum of Prefix Scores of Strings
    class Solution:
    def sumPrefixScores(self, words: List[str]) -> List[int]:
        # Build the trie structure from the list of words
        trie = self.buildTrie(words)
        # Calculate and return the prefix scores for each word
        return self.calculatePrefixScores(trie, words)

    def buildTrie(self, words: List[str]) -> Dict:
        trie = {}
        for word in words:
            node = trie
            for char in word:
                # Navigate through or create nodes in the trie
                node = node.setdefault(char, {})
                # Count occurrences of the prefix
                node['$'] = node.get('$', 0) + 1
        return trie

    def calculatePrefixScores(self, trie: Dict, words: List[str]) -> List[int]:
        scores = []
        for word in words:
            node = trie
            total_score = 0
            for char in word:
                # Move to the next node and accumulate the score
                node = node[char]
                total_score += node['$']
            scores.append(total_score)
        return scores


    # My calendar 2
    class MyCalendarTwo:

    def __init__(self):
        self.single_booked = []
        self.double_booked = []
        
    def intersection(self, intervals, s, e):

        l = bisect.bisect_left(intervals, s)
        r = bisect.bisect_right(intervals, e)
        
        intersection = []
        
        if l % 2:
            if intervals[l] != s:
                intersection.append(s)
            else:
                l = l + 1

        intersection += intervals[l:r]

        if r % 2:
            if intervals[r-1] != e:
                intersection.append(e)
            else:
                intersection.pop()

        return intersection
        
    def add(self, intervals, s, e):

        l = bisect.bisect_left(intervals, s)
        r = bisect.bisect_right(intervals, e)
        
        new = []
        if not l % 2:
            new.append(s)
            
        if not r % 2:
            new.append(e)

        intervals[l:r] = new

    def book(self, start: int, end: int) -> bool:

        if self.intersection(self.double_booked, start, end):
            return False
        
        intersection = self.intersection(self.single_booked, start, end)

        if intersection:
            for i in range(len(intersection) // 2):
                i1 = intersection[2*i]
                i2 = intersection[2*i+1]
                self.add(self.double_booked, i1, i2)

        self.add(self.single_booked, start, end)

        return True


        # All oned ata structure
        class AllOne:
    def __init__(self):
        self.myDict = {}

    def inc(self, key: str) -> None:
        if key in self.myDict:
            self.myDict[key] += 1
        else:
            self.myDict[key] = 1

    def dec(self, key: str) -> None:
        if key in self.myDict:
            if self.myDict[key] > 1:
                self.myDict[key] -= 1
            else:
                self.myDict.pop(key)

    def getMaxKey(self) -> str:
        if not self.myDict:
            return ""
        maxVal = max(self.myDict.values())
        for key in self.myDict.keys():
            if self.myDict[key] == maxVal:
                return key

    def getMinKey(self) -> str:
        if not self.myDict:
            return ""
        minVal = min(self.myDict.values())
        for key in self.myDict.keys():
            if self.myDict[key] == minVal:
                return key
    # Make Sum divisible by P
        def minSubarray(self, nums: List[int], p: int) -> int:
        totalSum = sum(nums)
        rem = totalSum % p

        if rem == 0:
            return 0

        prefixMod = {0: -1}
        prefixSum = 0
        minLength = len(nums)

        for i, num in enumerate(nums):
            prefixSum += num
            currentMod = prefixSum % p
            targetMod = (currentMod - rem + p) % p

            if targetMod in prefixMod:
                minLength = min(minLength, i - prefixMod[targetMod])

            prefixMod[currentMod] = i

        return minLength if minLength < len(nums) else -1


    # Divide Players into teams of Equal Skill
        def dividePlayers(self, skill: List[int]) -> int:
        freq=[0]*1001
        Sum, xMin, xMax=0, 1000, 1
        for x in skill:
            freq[x]+=1
            Sum+=x
            xMin=min(xMin, x)
            xMax=max(xMax, x)
        n_2=len(skill)//2
        if Sum%n_2!=0: return -1
        team_skill=Sum//n_2

        chemi=0
        l, r=xMin, xMax
        while l<r:
            fL, fR=freq[l], freq[r]
            if l+r!=team_skill or fL!=fR: return -1
            chemi+=fL*l*r
            l+=1
            r-=1
        if l==r and l*2==team_skill:
            chemi+=freq[l]//2*l*l
        return chemi
        

    # Minimum add ot make parenthesis valid
     def minAddToMakeValid(self, s: str) -> int:
        # Initialize the counter for minimum additions needed
        ans = 0
        
        # Initialize the balance of parentheses (open - close)
        bal = 0

        # Iterate through each character in the string
        for ch in s:
            if ch == '(':
                # If it's an opening parenthesis, increment the balance
                bal += 1
            else:
                # If it's a closing parenthesis, decrement the balance
                bal -= 1

            # If balance becomes negative (more closing than opening parentheses)
            if bal < 0:
                # Add the absolute value of balance to answer
                # This represents the number of opening parentheses we need to add
                ans += -bal
                # Reset balance to 0 since we've accounted for the imbalance
                bal = 0

        # After processing all characters, add any remaining open parentheses
        # This represents the number of closing parentheses we need to add
        ans += bal

        # Return the minimum number of additions needed to make the string valid
        return ans


        import heapq

class Solution:
    def smallestChair(self, times: List[List[int]], targetFriend: int) -> int:
        n = len(times)
        
        # Create a list of arrivals with friend index
        arrivals = [(times[i][0], i) for i in range(n)]
        
        # Sort friends by arrival time
        arrivals.sort()
        
        # Min-Heap to track available chairs
        availableChairs = list(range(n))
        heapq.heapify(availableChairs)

        # Priority queue to track when chairs are freed
        leavingQueue = []
        
        # Iterate through each friend based on arrival
        for arrivalTime, friendIndex in arrivals:
            # Free chairs that are vacated before the current arrival time
            while leavingQueue and leavingQueue[0][0] <= arrivalTime:
                heapq.heappush(availableChairs, heapq.heappop(leavingQueue)[1])
            
            # Assign the smallest available chair
            chair = heapq.heappop(availableChairs)
            
            # If this is the target friend, return their chair number
            if friendIndex == targetFriend:
                return chair
            
            # Mark the chair as being used until the friend's leave time
            heapq.heappush(leavingQueue, (times[friendIndex][1], chair))
        
        return -1  # Should never reach here


        # Minimum number of groups
            def minGroups(self, intervals: List[List[int]]) -> int:
        start_times = sorted(i[0] for i in intervals)
        end_times = sorted(i[1] for i in intervals)
        end_ptr, group_count = 0, 0

        for start in start_times:
            if start > end_times[end_ptr]:
                end_ptr += 1
            else:
                group_count += 1

        return group_count

    # Seperate Black and White Balls
    
    def minimum_steps(s):
        swaps = 0
        li = 0
        ri = len(s) - 1
        while li < ri:
            if s[li] == '1':
                if s[ri] == '0':
                    swaps += ri - li
                    li += 1
                ri -= 1
            else:
                if s[ri] == '1':
                    ri -= 1
                li += 1
        return swaps


    # Find the K-th character in string game 1
    def kthCharacter(self,k):
    s = "a"
        
    while len(s) < k:
        temp = ""
        for c in s:
            if c == 'z':
                temp += 'a'
            else:
                temp += chr(ord(c) + 1)
        s += temp
        
     return s[k - 1]


     # FLip equiavalent tree
         def flipEquiv(self, root1, root2):
        
        def checker(node1, node2):
            if not node1 and not node2:
                return True
            if not node1 or not node2 or node1.val != node2.val:
                return False
            return ((checker(node1.left, node2.left) or checker(node1.left, node2.right)) and
                    (checker(node1.right, node2.right) or checker(node1.right, node2.left)))
        
        return checker(root1, root2)


    # Remove subfolders
        def removeSubfolders(self, folder):
        """
        :type folder: List[str]
        :rtype: List[str]
        """

        # n is num of path, L is total length of all folder pahts
        # time complexity: O (n log n + L)
        # space complexity: O(L)

        folder.sort()
        res = []

        for path in folder:
            # check if res is empty or if the current path is not subfolder
            if not res or not path.startswith(res[-1] + '/'):
                # if it's True, append to res
                res.append(path)

        return res
        
    # Count square matrices
    class Solution:
    def countSquares(self, matrix: List[List[int]]) -> int:
        # Get dimensions of the matrix
        n = len(matrix)    # number of rows
        m = len(matrix[0]) # number of columns
        
        # Create a DP table with same dimensions as matrix
        dp = [[0] * m for _ in range(n)]
        
        # Variable to store total count of squares
        ans = 0
        
        # Initialize first column of DP table
        for i in range(n):
            dp[i][0] = matrix[i][0]
            ans += dp[i][0]
        
        # Initialize first row of DP table
        for j in range(1, m):
            dp[0][j] = matrix[0][j]
            ans += dp[0][j]
        
        # Fill the DP table for remaining cells
        for i in range(1, n):
            for j in range(1, m):
                # Only process if current cell in matrix is 1
                if matrix[i][j] == 1:
                    # Find minimum of left, top, and diagonal cells and add 1
                    dp[i][j] = 1 + min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1])
                ans += dp[i][j]
        
        return ans


        def minimumTotalDistance(self, robot: List[int], factory: List[List[int]]) -> int:
        robot.sort()
        factory.sort()
        m, n = len(robot), len(factory)
        dp = [[0]*(n+1) for _ in range(m+1)] 
        for i in range(m): dp[i][-1] = inf 
        for j in range(n-1, -1, -1): 
            prefix = 0 
            qq = deque([(m, 0)])
            for i in range(m-1, -1, -1): 
                prefix += abs(robot[i] - factory[j][0])
                if qq[0][0] > i+factory[j][1]: qq.popleft()
                while qq and qq[-1][1] >= dp[i][j+1] - prefix: qq.pop()
                qq.append((i, dp[i][j+1] - prefix))
                dp[i][j] = qq[0][1] + prefix
        return dp[0][0]

        # Circular sentence
            def isCircularSentence(self, sentence: str) -> bool:
        # Get the length of the sentence
        n = len(sentence)
        
        # First check: Compare first and last character of sentence
        # For a circular sentence, they must match
        if sentence[0] != sentence[n-1]:
            return False
            
        # Iterate through the sentence, starting from index 1 to n-2
        # We don't need to check first and last characters again
        for i in range(1, n-1):
            # When we find a space character
            if sentence[i] == ' ':
                # Check if the character before space (last char of current word)
                # matches the character after space (first char of next word)
                if sentence[i-1] != sentence[i+1]:
                    return False
                    
        # If we made it through all checks, the sentence is circular
        return True

    # Find if Array can be sorted
        def setbit(self, num):
        count = 0
        while num:
            count += num & 1
            num >>= 1
        return count

    def canSortArray(self, nums):
        n = len(nums)
        
        # Iterate and only swap adjacent elements with the same set bit count if needed
        for i in range(n - 1):
            for j in range(n - 1):
                if nums[j] > nums[j + 1] and self.setbit(nums[j]) == self.setbit(nums[j + 1]):
                    nums[j], nums[j + 1] = nums[j + 1], nums[j]
        
        # Check if the array is sorted
        return nums == sorted(nums)


    # Minimum Array End
        def minEnd(self, n: int, x: int) -> int:
        result = x
        remaining = n - 1
        position = 1
    
        while remaining:
            if not (x & position):
                result |= (remaining & 1) * position
                remaining >>= 1
            position <<= 1
    
        return result

    
# Shortest Sub Array with OR
def minimumSubarrayLength(self, nums, k):
    n = len(nums)
    bitCount = [0] * 32
    currentOR = 0
    left = 0
    minLength = float('inf')
    
    for right in range(n):
        currentOR |= nums[right]
        
        for bit in range(32):
            if nums[right] & (1 << bit):
                bitCount[bit] += 1
        
        while left <= right and currentOR >= k:
            minLength = min(minLength, right - left + 1)
            
            updatedOR = 0
            for bit in range(32):
                if nums[left] & (1 << bit):
                    bitCount[bit] -= 1
                if bitCount[bit] > 0:
                    updatedOR |= (1 << bit)
            
            currentOR = updatedOR
            left += 1
    
    return minLength if minLength != float('inf') else -1


# Moset beautiful item in a query
    def maximumBeauty(self, items: List[List[int]], queries: List[int]) -> List[int]:
        sorted_items = sorted(items, key=lambda item: item[0])  # Step 1: Sort items by price

        prices = [item[0] for item in sorted_items] # Step 2: Extract prices and beauties
        beauties = [item[1] for item in sorted_items]
        
        max_beauties = list(accumulate(beauties, max, initial=0)) # Step 3: Create running maximum beauty array
        
        result = []     # Step 4: Find maximum beauty for each query price
        for query_price in queries:
            index = bisect_right(prices, query_price)
            result.append(max_beauties[index])  
        return result

# Shortest SUbarray
    def findLengthOfShortestSubarray(self, arr):
        n = len(arr)
        left, right = 0, n - 1
        
        # Find longest non-decreasing suffix
        while right > 0 and arr[right - 1] <= arr[right]:
            right -= 1
        
        # If the entire array is already sorted
        if right == 0:
            return 0
        
        ans = right
        
        # Find the minimum length of subarray to remove
        while left < right and (left == 0 or arr[left - 1] <= arr[left]):
            while right < n and arr[left] > arr[right]:
                right += 1
            ans = min(ans, right - left - 1)
            left += 1
        
        return ans


    # Shortest Sub Array 3
        def shortestSubarray(self, nums, k):
        n = len(nums)
        prefixSum = [0] * (n + 1)
        
        for i in range(n):
            prefixSum[i + 1] = prefixSum[i] + nums[i]
        
        dq = deque()
        minLength = float('inf')
        
        for i in range(n + 1):
            while dq and prefixSum[i] - prefixSum[dq[0]] >= k:
                minLength = min(minLength, i - dq.popleft())
            
            while dq and prefixSum[i] <= prefixSum[dq[-1]]:
                dq.pop()
            
            dq.append(i)
        
        return minLength if minLength != float('inf') else -1


# Take Characters
def takeCharacters(self, text: str, req: int) -> int:
        freq = [0] * 3
        size = len(text)
        
        for char in text:
            freq[ord(char) - ord('a')] += 1
        
        left = 0
        right = 0
        
        if freq[0] < req or freq[1] < req or freq[2] < req:
            return -1
        
        for right in range(size):
            freq[ord(text[right]) - ord('a')] -= 1
            
            if freq[0] < req or freq[1] < req or freq[2] < req:
                freq[ord(text[left]) - ord('a')] += 1
                left += 1
        
        return size - (right - left + 1)


# Minimum obstacle removal to reach corner
def minimumObstacles(self, grid: List[List[int]]) -> int:
    #find the min cost path from (0,0) to (m-1, n-1)
    m = len(grid); n = len(grid[0])
    
    heap = [[grid[0][0], 0, 0]]
    visited = [[0]*n for _ in range(m)]
    visited[0][0] = 1
    while heap:
        c_cost, c_i, c_j = heappop(heap)
        if (c_i,c_j) == (m-1,n-1):
            return c_cost
        for n_i,n_j in [[c_i-1,c_j], [c_i,c_j+1], [c_i+1,c_j], [c_i,c_j-1]]:
            if 0<=n_i<m and 0<=n_j<n and not visited[n_i][n_j]:
                visited[n_i][n_j] = 1
                heappush(heap, [c_cost + grid[n_i][n_j], n_i, n_j])
    
    return -1