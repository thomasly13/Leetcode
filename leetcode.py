

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