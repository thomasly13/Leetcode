# Merge Intervals Recursive

def merge(intervals)
    #base case is if stack.length == 1 
 
     return intervals if intervals.length == 1
 
     entry = intervals.shift
 
     new_intervals = merge(intervals)
     temporary_interval = []
     stack = []
 
     new_intervals.each do |interval|
 
 
         num1 = entry[0]
         num2 = entry[-1]
 
         num3 = interval[0]
         num4 = interval[-1]
 
         array = [num1, num2, num3, num4]
 
         if ((num1 >= num3 && num1 <= num4) || (num2 >= num3 && num2 <= num4)) || ((num3 >= num1 && num3 <= num2) || (num4 >= num1 && num4 <= num2))
             min = array.min
             max = array.max 
             entry = [min, max]
         elsif (num4 < num2)
             stack.unshift(interval)
         else 
             entry = entry
             temporary_interval.push(interval)
         end    
     end
 
     stack.push(entry)
     stack.concat(temporary_interval)
     
     return stack 
     
 end

 # Merge Intervals Normal

 def merge(intervals)

    new_interval = intervals.sort

    stack = [new_interval.shift]

    new_interval.each do |interval|

        current_stack = {
            "start": stack[-1][0],
            "end": stack[-1][-1]
        }

        current_interval = {
            "start": interval[0],
            "end": interval[-1]
        }

        if (current_stack[:end] >= current_interval[:start]) && (current_stack[:end] <= current_interval[:end])
            stack[-1] = [current_stack[:start], current_interval[:end]]
        elsif (current_stack[:end] < current_interval[:start])
            stack.push(interval) 
        end
    end
    
    return stack
end


# Merge Strings Alternately 
#Javascript

# var mergeAlternately = function(word1, word2) {

#     create a flag that switches when the counters = the length of the words

#     let flag = false;

#     let final = "";

#     create two counters   
#     let counter1 = 0;
#     let counter2 = 0;

#     while(flag === false) {
#     if the counters are the same, word1[index counter1] gets added to a new string and +1 to counter1
#         if(counter1 === counter2){
#             final += word1[counter1]
#             counter1 += 1
#         } else if(counter1 === word1.length && counter2 !== word2.length){
#             final += word2.slice(counter2);
#             counter2 = word2.length;
#         } else if(counter2 === word2.length && counter1 !== word1.length) {
#             final += word1.slice(counter1);
#             counter1 = word1.length;
#         } else {
#             final += word2[counter2]
#             counter2 += 1
#         }


#         if(counter1 === word1.length && counter2 === word2.length) flag = true

#     if the counters are different word2[index counter2] gets added to a new string and +1 to counter2        

        
#     }
    
# return final

# };

# Baseball Game
def cal_points(operations)

    stack = []

    operations.each do |op|
        if op == "+"
            stack.push(stack[-1] + stack[-2])
        elsif op == "D"
            stack.push(stack[-1] * 2)
        elsif op == "C"
            stack.pop()
        else 
            stack.push(op.to_i)
        end
    end

    return stack.sum
    
end


# Valid Parenthesis
def is_valid(input)
    
    stack = []

    input.each_char do |parenthesis|

        if stack.length == 0 
            stack.push(parenthesis)

        elsif stack[-1] == "("
            if ((parenthesis != "{" && parenthesis != "[" && parenthesis != "(") && (parenthesis != ")"))
                return false
            elsif (parenthesis == ")")
                stack.pop()
            else 
                stack.push(parenthesis)
            end
            

        elsif stack[-1] == "{"
            if ((parenthesis != "(" && parenthesis != "[" && parenthesis != "{") && (parenthesis != "}"))
                return false
            elsif (parenthesis == "}")
                stack.pop()
            else 
                stack.push(parenthesis)
            end
            

        elsif stack[-1] == "["
            if ((parenthesis != "[" && parenthesis != "(" && parenthesis != "{") && (parenthesis != "]"))
                return false
            elsif (parenthesis == "]")
                stack.pop()
            else 
                stack.push(parenthesis)
            end

        end

    end

    return true if stack.length == 0 
    return false
end

# Greatest Common Divisor of Strings
def gcd_of_strings(str1, str2)
    return str1 if str1 == str2
    
    substrings = []

    biggest_divisor = ""

    str1 > str2 ? current_string = str2 : current_string = str1

    current_string.each_char.with_index do |char, index1|
        (index1...current_string.length).each do |index2|
            substring = current_string[index1..index2]
            next if (((str1.length / substring.length) != (str1.length / (substring.length.to_f))) || ((str2.length / substring.length) != (str2.length / (substring.length.to_f))))

            substrings.push(substring)
        end
    end

    substrings.each do |substring|
        length = biggest_divisor.length

        next if substring.length < length

        multiplier1 = str1.length / substring.length
        multiplier2 = str2.length / substring.length

        if ((substring * multiplier1 == str1) && (substring * multiplier2 == str2))
            biggest_divisor = substring
        end
    end

    return biggest_divisor
end

# Kids with the Greatest Number of Candies
def kids_with_candies(candies, extra_candies)
    max = candies.max 

    truthies = max - extra_candies

    final = []

    candies.each do |candy|
        if candy >= truthies 
            final.push(true)
        else 
            final.push(false)
        end
    end 

    return final
end

# Can Place Flowers
def can_place_flowers(flowerbed, n)
    counter = 0 
    
    flowerbed.each_with_index do |flower, index|

        if (flower == 0 && flowerbed[index - 1] == 0 && flowerbed[index + 1] == 0 && index != 0 && index != flowerbed.length - 1)
            counter += 1
            flowerbed[index] = 1
        elsif (index == 0 && flowerbed[index + 1] == 0 && flower == 0)
            counter += 1 
            flowerbed[index] = 1
        elsif (index == flowerbed.length - 1 && flowerbed[index - 1] == 0 && flower == 0)
            counter += 1 
            flowerbed[index] = 1
        end
    end 
    p counter

    return true if n <= counter 
    return false
end

# Asteroid Collision
def asteroid_collision(asteroids)

    flag = true
    
    while flag
    flag = false

    first = asteroids.shift
    first ? stack = [first] : stack = []

        asteroids.each_with_index do |asteroid, index|

            if stack.length == 0
                stack.push(asteroid)
                next
            end


            asteroid > 0 ? current_comparer = "positive" : current_comparer = "negative"
            stack[-1] > 0 ? current_top = "positive" : current_top = "negative"

    
            if (current_comparer == current_top) 
                stack.push(asteroid)
            elsif (current_top == "negative" && current_comparer == "positive")
                stack.push(asteroid)
            else 
                if (stack[-1].abs < asteroid.abs) 
                    stack[-1] = asteroid
                    flag = true
                elsif (stack[-1].abs == asteroid.abs)
                    stack.pop
                end
            end

        end    

    asteroids = stack
    end
    return stack

    
end

#Reverse Vowels of a String
def reverse_vowels(s)

    vowels = "aeiouAEIOU"

    current_vowels = [] 

    s.each_char.with_index do |char, index|
         
        if vowels.include?(char)
            current_vowels.push(char)
            s[index] = "*"
        end
    end

    s.each_char.with_index do |char, index|
        if char == "*"
            s[index] = current_vowels.pop
        end
    end

    return s
end

# Reverse Words in a String
def reverse_words(s)
    start_char = s.length - 1
    end_char = s.length - 1
    word = ""

    space = " "

    final_string = []

    until (start_char < 1)


        if(s[start_char] == space)
            word = s[start_char + 1..end_char]
            final_string.push(word) if word != space && word != ""
            word = ""
            start_char -= 1
            end_char = start_char
        else 
            start_char -= 1
        end
    end


    if s[start_char] == " "
        word = s[start_char + 1..end_char]   
    else 
        word = s[start_char..end_char]
    end



    final_string.push(word) if word != space && word != ""

    return final_string.join(" ")
    
end

# Find the difference of Two Arrays
def find_difference(nums1, nums2)

    final = [] 

    nums1.each do |num|
        
        if (!nums2.include?(num)) 
            if final[0]
                final[0].push(num) if !final[0].include?(num)
            else 
                final.push([num])
            end
        end
    end

    final.push([]) if !final[0]

    nums2.each do |num|
        
        if (!nums1.include?(num)) 
            if final[1]
                final[1].push(num) if !final[1].include?(num)
            else 
                final.push([num])
            end
        end
    end

    final.push([]) if !final[1]


    return final 
end

# Decode String
def decode_string(s)    

    flag = true

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz[]"

    while flag
        flag = false

        stack = ""
    
        multiplier = ""

        bracket_counter = 0
        
        string = ""

        
        s.each_char.with_index do |char, index|
            if (char == "[")
                if (bracket_counter != 1)
                    string += char
                    flag = true
                end
            elsif (char =="]")
                if(bracket_counter != 1)
                    bracket_counter -= 1
                    string += char
                else 
                    p "hitting"
                    stack += string * multiplier.to_i
                    bracket_counter = 0
                    string = ""
                    multiplier = ""
                end
            elsif alphabet.include?(char)
                if bracket_counter == 0 
                    stack += char
                else 
                    string += char                
                end

            else
                if bracket_counter > 0
                    if alphabet.include?(s[index + 1])
                        bracket_counter += 1
                    end
                    string += char
                else
                    if alphabet.include?(s[index + 1])
                        multiplier += char
                        bracket_counter += 1    
                    else 
                        multiplier += char
                    end

                end

            end
        end

        s = stack
    end

    return stack
    
end

# Two Sum

def two_sum(nums, target)

    tracker = {} 

    nums.each_with_index do |num, index|
        if tracker[num]
            return [tracker[num], index]
        else 
            tracker[target - num] = index
        end
    end

end

# Unique Number of Occurrences 
def unique_occurrences(arr)
    counter = Hash.new(0)

    checker = {}

    arr.each do |num|
        counter[num] += 1
    end

    counter.each do |key, value|
        if checker[value]
            return false
        else 
            checker[value] = key 
        end 
    end
    return true
end

# Determine if Two Strings Are Close 
def close_strings(word1, word2)

    return false if word1.length != word2.length 

    word1_count = Hash.new(0)

    word2_count = Hash.new(0)

    common_count = {}

    word1.each_char do |char|
        word1_count[char] += 1
    end

    word2.each_char do |char|
        word2_count[char] += 1
    end

    return true if word1_count == word2_count

    return false if word1_count.keys.sort != word2_count.keys.sort

    word1_count.each do |key, value|
        word2_count.each do |key2, value2|
            if value == value2 
                word1_count.delete(key)
                word2_count.delete(key2)
                break
            end
        end
    end

    return false if (word1_count != word2_count)

    return true

end

# Equal Row and Column Pairs
def equal_pairs(grid)
    size = grid.length 
    
    counter = 0 

    row_tracker = {}

    col_tracker = {}

    grid.each_with_index do |subarray, index|
        row_tracker[index] = subarray
    end

    size.times do |index1|
        subarray = []
        size.times do |index2|
            subarray.push(grid[index2][index1])
        end
        col_tracker[index1] = subarray
    end

    size.times do |index1|
        size.times do |index2|
            if col_tracker[index2] == row_tracker[index1]
                counter += 1
            end
        end
    end

    return counter
end

#Best Time to Buy and Sell Stock
def max_profit(prices)

    biggest_profit = 0

    min = prices.min 

    max = prices.max 


    if (prices.find_index(min) < prices.find_index(max))
        return max - min 
    end

    pivot = prices[0]

    prices[1..-1].each_with_index do |num, i|
        if (num < pivot)
            pivot = num
            next 
        else 
            sale = num - pivot
            if sale > biggest_profit 
                biggest_profit = sale
            end
        end
    end

    return biggest_profit

end

# Merge Two Sorted Lists
def merge_two_lists(list1, list2)
    
    merged_list = []


    until list1 == nil && list2 == nil

        current_node1 = list1

        current_node2 = list2

        if list1 == nil && list2 != nil
            merged_list.push(list2.val)
            list2 = list2.next
        elsif list1 != nil && list2 == nil
            merged_list.push(list1.val)
            list1 = list1.next
        elsif (current_node1.val < current_node2.val)
            merged_list.push(list1.val)
            list1 = list1.next
        else 
            merged_list.push(list2.val)
            list2 = list2.next
        end
    end


    merged_list
end

#Valid Palindrome
def is_palindrome(s)
    
    capital_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    lower_alphabet = "abcdefghijklmnopqrstuvwxyz"

    numbers = "0123456789"

    forward_string = ""
    backwards_string = ""
    
    s.each_char do |char|
        if capital_alphabet.include?(char)
            forward_string += char.downcase
            backwards_string = char.downcase + backwards_string
        elsif lower_alphabet.include?(char) || numbers.include?(char)
            forward_string += char
            backwards_string = char + backwards_string
        end
    end

    return true if forward_string == backwards_string

    false
end

# Time to sell stock 2
def max_profit(prices)

    total_profit = 0

    current_profit = 0
    
    pivot = prices[0]

    prices[1..-1].each_with_index do |sale, index|
        if (sale < pivot)
            pivot = sale
            total_profit += current_profit   
            current_profit = 0
        else 
            if current_profit < sale - pivot 
                current_profit = sale - pivot   
                if (index == prices.length - 2)
                    total_profit += current_profit
                end
            else 
                pivot = sale 
                total_profit += current_profit   
                current_profit = 0      
            end

        end
    end

    return total_profit
end

# Invert Binary Tree
def invert_tree(root)

    return root if root == nil

    new_node = TreeNode.new(root.val)

    if root.left == nil && root.right == nil
        return root
    end

    if root.left
        new_node.right = invert_tree(root.left)
    end

    if root.right
        new_node.left = invert_tree(root.right)
    end

    return new_node

end

# Valid Anagram
def is_anagram(s, t)

    return false if s.length != t.length 

    string1_counter = Hash.new(0)

    string2_counter = Hash.new(0)

    s.each_char do |char|
        string1_counter[char] += 1
    end

    t.each_char do |char|
        string2_counter[char] += 1
    end

    return string1_counter == string2_counter
end

# Binary Search 
def search(nums, target)

    return -1 if target < nums[0] || target > nums[-1]

    mid_index = nums.length / 2

    return -1 if nums[mid_index] != target && nums.length <= 1 

    return mid_index if nums[mid_index] == target

    smaller_array = nums[0...mid_index]

    larger_array = nums[mid_index + 1..-1]

    if (target < nums[mid_index])
        return search(smaller_array, target)
    else
        results = search(larger_array, target)
        if results == -1 
            return -1
        else 
            return results + mid_index + 1
        end
    end
    
end

# letter combinations

#var letterCombinations = function(digits) {
    #let keypad = {
        #2: ["a", "b", "c"],
        #3: ["d", "e", "f"],
        #4: ["g", "h", "i"],
        #5: ["j", "k", "l"],
        #6: ["m", "n", "o"],
        #7: ["p", "q", "r", "s"],
        #8: ["t", "u", "v"],
        #9: ["w", "x", "y", "z"]
    #}
    #let results = [];
    #if (digits.length === 0) return [];
    #let array = digits.split("");
    #let letters = keypad[array.shift()];
    #let newArray = letterCombinations(array.join(""));
    #for (let i = 0; i < letters.length; i++) {
        #if (newArray.length === 0) {
            #results.push(letters[i])
        #} else {
            #for (let j = 0; j < newArray.length; j++) {
                #results.push(letters[i] + newArray[j])
            #}
        #}
    #}
    #return results;
#};

#Flood Fill
def flood_fill(image, sr, sc, color)

    starting_color = image[sr][sc]
    return image if starting_color == color
    image[sr][sc] = color

    
    queue = [[sr, sc]]

    while queue.length > 0 
        current_square = queue.shift

        current_x = current_square[1]
        current_y = current_square[0]

        adjacent = []

        adjacent.push([current_y - 1, current_x]) if current_y != 0
        adjacent.push([current_y, current_x - 1]) if current_x != 0
        adjacent.push([current_y + 1, current_x]) if current_y != image.length - 1
        adjacent.push([current_y, current_x + 1]) if current_x != image[0].length - 1

        adjacent.each do |coordinates|
            x = coordinates[1]
            y = coordinates[0]

            if image[y][x] == starting_color
                image[y][x] = color
                queue.push(coordinates)
            end
        end
    end

    return image
end

# reversed linked list 
def reverse_list(head)
    return head if head == nil || head.next == nil

    current_node = head
    next_node = nil

    while true
        if current_node.next == nil
            current_node.next = next_node
            return current_node
        end
        prev_node = current_node.next
        current_node.next = next_node
        next_node = current_node
        current_node = prev_node
    end
end

# leaf similar tree
def leaf_similar(root1, root2)
    dfs(root1) == dfs(root2)
end

def dfs(root)
    return [root.val] if !root.left && !root.right
    root.left ? left = dfs(root.left) : left = []
    root.right ? right = dfs(root.right) : right = []
    return left.concat(right)
end

# Is Subsequence

def is_subsequence(s, t)

    pointer = 0

    t.each_char do |char|
        if s[pointer] == char
            pointer += 1
        end
        return true if pointer == s.length
    end

    return pointer == s.length


end

# conatainer with most water
def max_area(height)
    
    # o(n)

    # have two pointers starting at the middle going out to 0 and numOfTowers, finding the max area 

    num_of_towers = height.length # 9 

    pointer_b = num_of_towers - 1 # 4
    pointer_a = 0  # 3 

    greatest_area = 0  # 12


    while (pointer_a != pointer_b)

        width = pointer_b - pointer_a 

        tower_a = height[pointer_a]
        tower_b = height[pointer_b]

        (tower_a > tower_b) ? length = tower_b : length = tower_a
       
        area = width * length 
        greatest_area = area if area > greatest_area 

        if tower_a > tower_b 
            pointer_b -= 1
        else tower_a < tower_b 
            pointer_a += 1         
        end
    end

    return greatest_area 
end

# Max Number of K-Sum Pairs

def max_operations(nums, k)
    sorted_array = nums.sort

    pointer1 = 0

    pointer2 = sorted_array.length - 1

    operationCounter= 0 

    while (pointer1 < pointer2) 
        first_number = sorted_array[pointer1]
        second_number = sorted_array[pointer2]

        if (first_number + second_number == k)
            pointer1 += 1
            pointer2 -= 1
            operationCounter += 1
        elsif( first_number + second_number < k)
            pointer1 += 1
        else 
            pointer2 -= 1
        end
    end

    return operationCounter
end

# Counting Bits
def count_bits(n)
    #create an array to push values into
    ans = [0, 1]

    #set base cases
    return [0] if n == 0
    return ans if n == 1

    #set a pointer
    i = 2

    temp = [1]

    #while loop for while pointer is not greater than n
    while i <= n
        # calculate number of 1's in binary representation
        ans.concat(temp.concat(temp.map { |num| num + 1 }))
        i *= 2
    end

    ans.pop(i - n - 1)
    
    # return ans array
    return ans
end

# Maximum Twin Sum of a Linked List
def pair_sum(head)
    list_length = linked_list_length(head) - 1

    max_sum = 0

    index = 0

    tracker = {}

    current_node = head

    while current_node
        if tracker[list_length - index]
            current_sum = tracker[list_length - index] + current_node.val
            max_sum = current_sum if current_sum > max_sum
        else
            tracker[index] = current_node.val
        end
        index += 1
        current_node = current_node.next
    end
    
    return max_sum
end                                                   

def linked_list_length(head)
    
    current_node = head
    
    counter = 0 
    
    while current_node
        counter += 1
        current_node = current_node.next
    end
    
    counter
end

# Odd Even Linked List
def odd_even_list(head)
    # create two set of variables, odd and even

    # keep track of index 

    # for whatever it is, make the current node the next and change the current node 

    return head if (head == nil || head.next == nil)

    index = 1

    current_node = head.next.next

    odd_node = head

    current_odd_node = head

    even_node = head.next

    current_even_node = head.next

    while current_node 
        if (index % 2 == 1)
            current_odd_node.next = current_node 
            current_odd_node = current_node 
        else 
            current_even_node.next = current_node
            current_even_node = current_node
        end

        index += 1
        current_node = current_node.next

    end

    current_even_node.next = nil
    current_odd_node.next = even_node 
    return odd_node

    
end

# Delete the Middle Node of a Linked List

def delete_middle(head)

    return nil if head.next == nil

    if head.next.next == nil
        head.next = nil
        return head
    end

    fast_pointer = 1 

    connector_node_1 = head

    connector_node_2 = head.next.next

    fast_node = head.next.next.next

    while fast_node 
        fast_node = fast_node.next 
        fast_pointer += 1

        if (fast_pointer % 2 == 0)
            connector_node_1 = connector_node_1.next
            connector_node_2 = connector_node_2.next
        end

    end

    connector_node_1.next = connector_node_2

    return head
    

    
end

# Find the Highest Altitude
def largest_altitude(gain)
    current_altitude = 0 

    highest_altitude = 0

    gain.each do |change|
        current_altitude += change
        highest_altitude = current_altitude if highest_altitude < current_altitude
    end

    return highest_altitude
end

# Maximum Number of Vowels in a Substring of Given Length
def max_vowels(s, k)
    
    vowels = "aeiou"

    pointer1 = 0 

    pointer2 = 0

    current_vowel_count = 0

    biggest_vowel_count = 0

    until (pointer2 - pointer1 == k)
        if (vowels.include?(s[pointer2]))
            current_vowel_count += 1
        end
        pointer2 += 1
    end

    pointer2 -= 1

    until (pointer2 == s.length)

        return current_vowel_count if current_vowel_count == k

        biggest_vowel_count = current_vowel_count if current_vowel_count > biggest_vowel_count

        current_vowel_count -= 1 if vowels.include?(s[pointer1])

        pointer1 += 1

        pointer2 += 1

        current_vowel_count += 1 if pointer2 != s.length && vowels.include?(s[pointer2]) 

        biggest_vowel_count = current_vowel_count if current_vowel_count > biggest_vowel_count
    end
    
    return biggest_vowel_count
end

#Linked list cycle
def hasCycle(head)
    # hash with node as keys 

    tracker = {} #1, 2

    current_node = head # 1

    while current_node 
        
        if tracker[current_node]
            return true 
        end
        
        tracker[current_node] = 1 
        current_node = current_node.next

    end 

    return false
end

# Binary Tree Inorder Traversal
def inorder_traversal(root)

    return [] if root == nil
    
    left = inorder_traversal(root.left)
    right = inorder_traversal(root.right)


    return left.concat([root.val], right)
end

# Binary Tree Preorder Traversal
def preorder_traversal(root)
    return [] if root == nil

    left = preorder_traversal(root.left)
    right = preorder_traversal(root.right)

    return [root.val].concat(left, right)
end

# Binary Tree Postorder Traversal
def postorder_traversal(root)
    return [] if root == nil

    left = postorder_traversal(root.left)
    right = postorder_traversal(root.right)

    return left.concat(right, [root.val])
end

# Pivot Index
def pivot_index(nums)
    # Create two sliding windows

    # create a pivot

    # depending on which one is bigger change then the pivot 

    # save pivots to a hash, if it exist return -1 

    # if pivot goes negative ot to length return -1

    size = nums.length
    
    pivot = 0 

    sliding_window_1 = 0

    sliding_window_2 = nums[1..-1].sum

    while pivot < size - 1

        return pivot if sliding_window_1 == sliding_window_2
        
        sliding_window_1 += nums[pivot]

        pivot += 1

        sliding_window_2 -= nums[pivot]

    end

    if sliding_window_1 == sliding_window_2 
        return pivot
    else
        return -1
    end
end

# Search Matrix
def search_matrix(matrix, target) # [1... , 10..., 23...], 3
    # look at the first integer of every row until we find the row 

    # looking at the row, do a binary search to find the number or return false 

    flattened = matrix.flatten

    return binary_search(flattened, target)

end

def binary_search(array, target)

    # base case of checking if the array is length 1 or lower, and checking the value 

    # find middle index 

    # return true if middle value = target 

    # lower_array = numbers smaller than middle value 

    # higher_array = numbers bigger than middle value 

    # call this recursively depending on if the target is bigger or smaller than the middle value 

     # [1, 3], 3

    return false if array.length <= 1 && array[0] != target 

    middle_index = array.length / 2 # 1

    middle_value = array[middle_index] #3

    return true if middle_value == target 

    lower_array = array[0...middle_index] 

    higher_array = array[middle_index + 1..-1]

    if (target > middle_value) 
        if (binary_search(higher_array, target))
            return true
        end
    else 
        if (binary_search(lower_array, target))
            return true
        end
    end

    return false
end

#Median of two sorted arrays
def find_median_sorted_arrays(nums1, nums2)
    merged = []

    while nums1.length > 0 && nums2.length > 0
        if nums1[0] > nums2[0]
            merged.push(nums2.shift)
        else 
            merged.push(nums1.shift)
        end
    end

    merged.concat(nums1, nums2)
    size = merged.length

    size % 2 == 1 ? merged[size / 2] : (merged[size / 2] + merged[size / 2 - 1]) / 2.0
end

# Add Two Numbers from Linked List
def add_two_numbers(l1, l2)
    #iterate through both list and add the corresponding indexes

    array1 = []

    array2 = []

    current_node_1 = l1 

    current_node_2 = l2
    
    while current_node_1 
        array1.push(current_node_1.val)
        current_node_1 = current_node_1.next
    end
    
     while current_node_2 
        array2.push(current_node_2.val)
        current_node_2 = current_node_2.next
    end

    num1 = array1.reverse.join("").to_i
    num2 = array2.reverse.join("").to_i

    sum = num1 + num2

    sum_array = sum.to_s.split("").reverse

    sum_array = sum_array.map{|num| num.to_i}

    final = nil
    current_node = nil
    sum_array.each do |nums|
        if !final 
            final = ListNode.new(nums)
            current_node = final
        else
            current_node.next = ListNode.new(nums)
            current_node = current_node.next
        end
    end

    return final
    
end

# Sort Array by parity
def sort_array_by_parity(nums)

    pointer = 0 

    odd_array = [] 

    even_array = []

    nums.each do |num|
        if num % 2 == 1
            odd_array.push(num)
        else 
            even_array.push(num)
        end
    end

    return even_array.concat(odd_array)
end 

# Monotonic Array
def is_monotonic(nums)
    return true if nums.sort == nums || nums.sort == nums.reverse
    false
end

def is_monotonic(nums)
    return true if nums.length < 3
    increasing = nil
    nums[1..-1].each_with_index do |num, index|
        if num > nums[index]
            increasing = true
            break
        elsif num < nums[index]
            increasing = false
            break
        else
            next
        end
    end
    nums[2..-1].each_with_index do |num, index|
        if (increasing == true && num < nums[index + 1]) ||(increasing == false && num > nums[index + 1]) 
            return false 
        end
    end
    true
end

# Remove colored pieces if both neighbors are the same color
def winner_of_game(colors)

    return false if colors.length < 3

    player = "A"

    player_a_index = 0

    player_b_index = 0
    #AAABABB
    while true
        if player == "A"
            

            colors[player_a_index..-1].each_char.with_index do |char, index|
                next if player_a_index + index == 0
                if player_a_index + index == colors.length - 1
                    return false
                end
                if char == player 
                    if colors[player_a_index + index - 1] == char && colors[player_a_index + index + 1] == char
                        colors[player_a_index + index] = ""

                        player = "B"

                        player_a_index += index - 1
                        
                        break
                    end
                end
            end

        else
            colors[player_b_index..-1].each_char.with_index do |char, index|
                next if player_b_index + index == 0
                if player_b_index + index == colors.length - 1
                    return true
                end
                if char == player 
                    if colors[player_b_index + index - 1] == char && colors[player_b_index + index + 1] == char
                        colors[player_b_index + index] = ""

                        player = "A"

                        player_b_index += index - 1
                        
                        break
                    end
                end
            end
            
        end

         
    end
    
end

# Number of good pairs

def num_identical_pairs(nums)

    counter = Hash.new(0)

    total = 0

    nums.each do |num|
        counter[num] += 1
    end

    counter.each_value do |amount|
        if amount > 1
            total += calculation(amount)
        end
    end
    return total
end

def calculation(amount)
    return 1 if amount == 2

    return calculation(amount - 1) + (amount - 1)
end

# MErge two sorted list
def merge_two_lists(list1, list2)
    # holding two current nodes 

    # looping through both lists, comparing the current nodes and adding the lesser node 

    # loop ends when one of the current nodes hits nil and Ill add the rest of the other list to the end 

    if !list1
        return list2
    elsif !list2
        return list1
    end

    current_node_a = list1 # heads of the list => 2 [ 2, 4]

    current_node_b = list2 #heads of the list => 1 [1, 3, 4]

    current_final_node = nil

    if current_node_a.val > current_node_b.val
        final_head = current_node_b 
        current_node_b = current_node_b.next 
        current_final_node = final_head
    else
        final_head = current_node_a # [1]
        current_node_a = current_node_a.next
        current_final_node = final_head
    end
    # [1, 1, 2, 3, 4]
    # heads of the list => nil []
     #heads of the list => 4 [4]
    while (current_node_a && current_node_b)

        if current_node_a.val > current_node_b.val
            current_final_node.next = current_node_b
            current_final_node = current_final_node.next
            current_node_b = current_node_b.next 

        else
            current_final_node.next = current_node_a
            current_final_node = current_final_node.next
            current_node_a = current_node_a.next
        end

    end

    current_final_node.next =  current_node_a || current_node_b

    return final_head

    
end

# longest substring without repeating characters
def length_of_longest_substring(s)
    return 0 if s.length < 1
    
    pointer1 = 0
    pointer2 = 0
    
    array = s.split('')

    highest_count = 1

    current_sub = array[pointer1..pointer2]

    
    until pointer2 == s.length - 1
        pointer2 += 1
        current_comparer = array[pointer2]

        

        if current_sub.include?(current_comparer)
            highest_count = current_sub.length if highest_count < current_sub.length 
            flag = true
            while flag
                if array[pointer1] == current_comparer
                    flag = false
                    pointer1 += 1
                    current_sub = array[pointer1..pointer2]
                else
                    pointer1 += 1 
                end
            end
        else
            current_sub = array[pointer1..pointer2]
            highest_count = current_sub.length if highest_count < current_sub.length 
        end
    end

    return highest_count
end

# Majority element 2
def majority_element(nums)
    comparer = nums.length/3.0

    counter = Hash.new(0)

    tracker = Set.new()

    final_array = []

    nums.each do |num|
        counter[num] += 1
        if !tracker.include?(num) && counter[num] > comparer 
            final_array.push(num) 
            tracker.add(num)
        end
    end


    final_array
end

#Longest Common prefix
def longest_common_prefix(strs)

    first_word = strs[0]

    prefix = ""

    first_word.length.times do |index|
        pre = first_word[0..index]
        if strs.all? {|word| word[0..index] == pre}
            prefix = pre
        else
            return prefix
        end
    end
    return prefix
end

# Power of Four
def is_power_of_four(n)
    
    return true if n == 1
    
    return false if n < 4
    current_num = n

    while true 
        divided = current_num / 4.0 

        return false if divided != divided.floor

        return true if divided == 1

        current_num = divided
         
    end
end

# Search Insert Position
def search_insert(nums, target)
   

    nums.each_with_index do |num, index| 
        if num == target || num > target 
            return index
        end
    end

    return nums.length
end

# Length of last word
def length_of_last_word(s)
    return s.split(" ")[-1].length
end

# Added plus one
def plus_one(digits)
    stringify = digits.map {|num| num.to_s }

    new_num = stringify.join("").to_i

    return (new_num + 1).to_s.split("").map{|char| char.to_i}
end

# Square Root
def my_sqrt(x)
    current = 0
 
     while true 
         squared = current ** 2
         if squared == x 
             return current 
         elsif squared > x
             return current - 1
         end
         current += 1
     end
 end


# Remove Duplicates from Sorted List
def delete_duplicates(head)
    return head if !head || !head.next
    
    previous_node = head

    next_node = head.next

    while next_node 
        if previous_node.val == next_node.val
            previous_node.next = next_node.next
            next_node = previous_node.next
        else 
            previous_node = next_node
            next_node = next_node.next
        end
    end

    return head
end

# Remove Linked List Elements
def remove_elements(head, val)

    return head if !head 
    
    current_head = head 

    until current_head.val != val 
        if current_head.next
            current_head = current_head.next
        else
            return nil
        end
    end

    previous_node = current_head
    
    next_node = previous_node.next 
    
    until !next_node
        if next_node.val == val 
            previous_node.next = next_node.next
            next_node = previous_node.next

        else
            previous_node = next_node
            next_node = next_node.next
        end 
    end

    return current_head
end

# -Better compress string

# -Given a string of chars and string ints, where there can be duplicate chars, return an alphabetically ordered string with no repeat chars.

# ex: ‘a3b5c7a3b43’  →return ‘a6b48c7’
# ex: ‘b3d2a1c12b2a111’ → return ‘a112b5c12d2’

def compress(string)
	# time complexity == O(n)

	# space complexity == O(1)

	# ‘a3b5c7a3b4’
	current _letter = "" # “b”

	current_count = "" # “43”

	alphabet = “abcdefghijklmnopqrstuvwxyz”

	Counter = Hash.new(0) # {“” =>  0, “a” => 6 , “b” => 48, “c” => 7}
	
	Index = 0 # 11
	# string.length = 11 
	While index < string.length 
		
		If alphabet.includes?(string[index]) # “3”
			Counter[current_letter] += current_count.to_i
			Current_count = “”
			Current_letter = string[index]
		Else 
			Current_count += string[index] # “3”
		End
		Index += 1
	End

	Counter[current_letter] += current_count.to_i
	
	Final_string = “” # “a6b48c7”

	Alphabet.each_char do |char| # char = “a”
		If counter[char]
			Final_string += char + counter[char].to_str
		end
	End

	Return final_string   # “a6b48c7”
	



End
	# if it is a letter, add the count of the letter into the hash, and current letter variable to the new letter, and then reset the count variable to a “”

	#iterate until end of string 

	#iterate through my alphabet string, and then check if the hash has it, and if it does I’ll add it and the count to my final hash

All lowercase letters

combing any compressions 

Ordering the list in alphabetical order 

	# current letter variable 

	# current count variable 

	#  counter hash 

	# while loop to iterate through 

	# inside : check if it’s a letter 
	

	#return this final string 

    # Permutaions
    def permute(nums)
        # return [nums] if nums.length < 2 
        # amount of answer arrays is equal to, the length times the previous length 
    
        # shallow copy of the previous(recurisve)
    
        # an array for final return
    
        # pointer = 0 
    
        # until looop for pointer and length 
    
        # iterate through my current arrays and insert using the pointer 
       
        return [nums] if nums.length == 1
        added_number = nums.pop() #
    
        new_array = permute(nums) 
    
        return_array = [] 
    
        pointer = 0 # 3
      
        until pointer == new_array[0].length + 1
            new_array.each do |subarray| # [0, 1]
                dup_array = subarray.dup()
                return_array.push(dup_array.insert(pointer, added_number))
            end
            pointer += 1
        end
    
        return return_array
    
    
    
    end

    # @param {Integer[]} nums1
# @param {Integer[]} nums2
# @return {Float}
def find_median_sorted_arrays(nums1, nums2)

    pointer1 = 0
    pointer2 = 0

    median = []
    size = nums1.length + nums2.length

    counter = 0

    size % 2 == 1 ? target = size / 2 : target = size / 2.0

    if nums1.length == 0 
        if target.is_a? Float
            median = [nums2[target], nums2[target - 1]]
            return median.sum / 2.0
        else
            return nums2[target]
        end
    elsif nums2.length == 0
        if target.is_a? Float
            median = [nums1[target], nums1[target - 1]]
            return median.sum / 2.0
        else
            return nums1[target]
        end
    end
    while median.length != 2
        if target.is_a? Float
            if counter == target || counter == target - 1
            if nums1[pointer1] < nums2[pointer2]
                number = nums1[pointer1]
                if pointer1 == nums1.length - 1
                    nums1[pointer1] = 1000000
                else
                    pointer1 += 1
                end
            else
                number = nums2[pointer2]
                if pointer2 == nums2.length - 1
                    nums2[pointer2] = 1000000
                else
                    pointer2 += 1
                end
            end
            median.push(number)

            else
                if nums1[pointer1] < nums2[pointer2]
                    if pointer1 == nums1.length - 1
                        nums1[pointer1] = 1000000
                    else
                        pointer1 += 1
                    end
                else
                    if pointer2 == nums2.length - 1
                        nums2[pointer2] = 1000000
                    else
                        pointer2 += 1
                    end
                end
                counter += 1
            end
        else
            if counter == target
                return nums1[pointer1] < nums2[pointer2] ? nums1[pointer1] : nums2[pointer2]
            else
                if nums1[pointer1] < nums2[pointer2]
                    if pointer1 == nums1.length - 1
                        nums1[pointer1] = 1000000
                    else
                        pointer1 += 1
                    end
                else
                    if pointer2 == nums2.length - 1
                        nums2[pointer2] = 1000000
                    else
                        pointer2 += 1
                    end
                end
                counter += 1
            end
        end
    end
    
    return median.sum / 2.0
end

def roman_to_int(s)
    counter = 0
    split = s.split('')
    split.each do |char|
    puts char
        if char == 'I'
        counter += 1
        elsif char == 'V' 
        counter += 5
        elsif char == 'X' 
        counter += 10
        elsif char == 'L' 
        counter += 50
        elsif char == 'C' 
        counter += 100
        elsif char == 'D' 
        counter += 500
        elsif char == 'M' 
        counter += 1000
        end
    end
    if s.include?('IV')
    counter -= 2
    end
    if s.include?('IX')
    counter -= 2
    end
    if s.include?('XL')
    counter -= 20
    end
    if s.include?('XC')
    counter -= 20
    end
    if s.include?('CD')
    counter -= 200
    end
    if s.include?('CM')
    counter -= 200
    end


    return counter
end

# Added numbers of laser beams
def number_of_beams(bank)

    # total amount of lasers
    laserbeams = 0 
    
    return 0 if bank.length == 0

    # have a current row variable
    current_row_index = 0
    current_row = bank[0]

    # and a next row variables
    comparer_row_index = 1
    comparer_row = bank[1]

    until comparer_row_index == bank.length
        comparer_lasers = 0

        comparer_row.each_char do |column|
            comparer_lasers += 1 if column == '1'
        end

        if comparer_lasers == 0
            comparer_row_index += 1
            comparer_row = bank[comparer_row_index]
            next
        end

        current_lasers = 0
        
        current_row.each_char do |column|
            current_lasers += 1 if column == '1'
        end

        if current_lasers == 0
            current_row_index = comparer_row_index
            comparer_row_index += 1

            current_row = bank[current_row_index]
            comparer_row = bank[comparer_row_index]
            next
        end

        laserbeams += current_lasers * comparer_lasers

        current_row_index = comparer_row_index
        comparer_row_index += 1

        current_row = bank[current_row_index]
        comparer_row = bank[comparer_row_index]

    end


    return laserbeams
    
end

# Minimum Number of Operations to Make Array Empty
def min_operations(nums)

    counter = Hash.new(0)

    nums.each do |num|
        counter[num] += 1
    end

    operations = 0

    counter.each_key do |key|
        return -1 if counter[key] == 1

        current_key = counter[key]

        until current_key == 0
            if current_key > 4 
                operations += 1
                current_key -= 3
            elsif current_key % 2 == 0
                operations += 1
                current_key -= 2
            else
                operations += 1
                current_key -= 3
            end
        end
    
    end

    return operations



end

# Longest Increasing Subsequence
def length_of_lis(nums)
    dp = [1] * (nums.length)

    (0...nums.length).each do |i|
        (0...i).each do |j|
            dp[i] = [dp[i], 1 + dp[j]].max if nums[i] > nums[j]
        end
    end
    dp.max
end


# Maximum difference between node adn ancestor
def max_ancestor_diff(root)
    return 0 unless root
    res = 0
    stack = []
    dfs = -> node do
        val = node.val
        stack << val
        left = node.left
        right = node.right
        dfs.(left) if left
        dfs.(right) if right
        stack.pop
        stack.each do |anc|
            res = [res, (anc - val).abs].max
        end
    end

    dfs.(root)
    res
end


# 3sum
def three_sum(nums)
    output = []
    return output if nums.length < 3
  
    nums.sort!
  
    nums.each_with_index do |num, index|
      # If we have same element on left of the index,
      # then it will give same subset again... [-3,-3,1,2]
      # In this case, after first iteration we will get a subset [-3,1,2] and
      # In 2nd we must not make a duplicate subset
      # we have this checker num == nums[index - 1], if matches will go to next iteration
      next if index > 0 && num == nums[index - 1]
  
      # the next number of num on left, and last number on right
      left = index + 1
      right = nums.length - 1
      # sorted array, so loop will run till right is higher number
      while left < right
        # We will take the summation of 3 of them
        summation = nums[index] + nums[left] + nums[right]
        # if 0, then we will save the subset in output array
        if summation == 0
          output << [nums[index], nums[left], nums[right]]
          # We have to avoid duplicate next value, so we will increase left index by 1
          # and we have to check if the previous index matched and also the right index is higher
          # if both TRUE then we will increase left by 1 again
          left += 1
          left += 1 while nums[left] == nums[left - 1] && left < right
  
          # if summation is smaller than zero, increase left index, cz array is sorted
        elsif summation < 0
          left += 1
          # if summation is higher than zero, decrease right index, cz array is sorted
        elsif summation > 0
          right -= 1
        end
      end
    end
    output
  end

  #Minimum NUmber of steps to make two strings anagram
  def min_steps(s, t)
    h = Hash.new(0)
    s.each_char { |ch| h[ch] += 1 }
  
    count = 0
  
    t.each_char do |ch|
      if h[ch] > 0
        h[ch] -= 1
      else
        count += 1
      end
    end
  
    count
  end

  #Insert Delete GetRandom O(1)
  class RandomizedSet
    def initialize()
        @randomized_set = Hash.new
    end


=begin
    :type val: Integer
    :rtype: Boolean
=end
    def insert(val)
        flag = true 
        if @randomized_set.has_key?(val)
            flag = false 
        else
            @randomized_set[val] = true 
        end 
        flag 
    end


=begin
    :type val: Integer
    :rtype: Boolean
=end
    def remove(val)
        flag = false 
        if @randomized_set.has_key?(val)
            flag = true 
            @randomized_set.delete(val)           
        end 
        flag 
    end


=begin
    :rtype: Integer
=end
    def get_random()
        random = rand(0...@randomized_set.length)        
        @randomized_set.keys[random]
    end


end

#Unique number of occurences
def unique_occurrences(arr)
    counter = Hash.new(0)

    checker = {}

    arr.each do |num|
        counter[num] += 1
    end

    counter.each do |key, value|
        if checker[value]
            return false
        else 
            checker[value] = key 
        end 
    end
    return true
end

# Longest Palindrome Substring
def longest_palindrome(s)
    str = ""
    ar = s.split("")
    rev = s.reverse
    count = 0
    (0..ar.length-1).each do |i|
        elem = ar[i]
        subs = check_palindrome(s, elem, i)
        if subs.length > str.length
            str = subs
        end
    end
    str
end
def check_palindrome s,elem,i
    str = ""
    count = 0
    rinx = s.rindex(elem)
    while( str.length == 0)
        substr = s.slice(i, rinx-i+1)
        if substr == substr.reverse && substr.length > count
            count = substr.length
            str = substr
        end
        break if rinx == 0
        s = s.slice(0, rinx)
        rinx = s.rindex(elem)
    end
    str
end

# integer to Roman
def int_to_roman(num)
    # basic_int_roman_map = {
    #     "I": 1,
    #     "IV": 4,
    #     "V": 5,
    #     "IX": 9
    #     "X": 10
    #     "LX": 40,
    #     "L": 50,
    #     "XC": 90,
    #     "C": 100,
    #     "CD": 400,
    #     "D": 500,
    #     "CM": 900,
    #     "M": 1000,
    # }
    alphas = %w[I IV V IX X XL L XC C CD D CM M]
    alpha_values = [1, 4, 5, 9, 10, 40, 50, 90, 100, 400, 500, 900, 1000]
    ans = ""
    alphas.zip(alpha_values).reverse.each_with_index do |alpha_value, index|
      while num >= alpha_value[1]
        ans << alpha_value[0]
        num -= alpha_value[1]
      end
    end
    ans
end

#Search in a rotated sorted array
def search(nums, target)
    l = 0
    r = nums.size - 1

    while l <= r
      mid = (l + r) / 2
      guess = nums[mid]

      return mid if guess == target
        
      if guess >= nums[l]
        if nums[mid] >= target && target >= nums[l]
          r = mid - 1
        else
          l = mid + 1
        end
      else
        if nums[mid] <= target && target <= nums[r]
          l = mid + 1
        else
          r = mid - 1
        end
      end
    end

    -1
end

# Permutations 2
def permute_unique(nums)
    freq = nums.tally 
    # {1 => 2, 2 => 1}
    res = []

    p_helper(freq.keys, freq, 0, [], res, nums.length )

    res 
end

def p_helper(arr, freq, i, slate, res, n )
    if i == n
        res << slate.dup
        return 
    end 

    arr.each do |num|
        next if freq[num] == 0 
        slate << num 
        freq[num] -= 1
        p_helper(arr, freq, i+1, slate, res, n )
        freq[num] += 1
        slate.pop 
    end 
end 

# Rotate image
def rotate(matrix)
    for i in (0...matrix.size)
        for j in ((i + 1)...matrix.size)
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        end
    end

    matrix.each &:reverse!
end

# Stairs
def climb_stairs(n)
    ways = 1 # Step 0, 1 way.
    w1, w2 = 0,0 # Step 1 and 2, not calcutated.
    (1..n).each do |i|
      w1, w2 = w1 + ways, w2 + ways
      ways, w1 = w1, w2
      w2 = 0
    end
    return ways
  end


# Minimum Falling Path Sum

def min_falling_path_sum(matrix)
    n = matrix.length
    (n - 1).downto(1) do |i|
        matrix[i - 1] = matrix[i - 1].map.with_index do |a, j|
            a + matrix[i][
                [0, j - 1].max .. [n - 1, j + 1].min
            ].min
        end
    end
    matrix.first.min
end

# Maximum Length of a concatenated string with unique characters
def max_length(arr)
    i = 0
    selected=Array.new(26,0)
    len = 0
    return help(i, arr, selected, len)    
end
def help(i, arr, selected, len)
    return len if i == arr.length
    currString = arr[i]
    p compare(selected, currString)
    if (compare(selected, currString) == false)
        help(i+1, arr, selected, len)
    else
        # pick
        for j in (0...currString.length)
            selected[currString[j].ord - "a".ord]=1
        end
        len += currString.length
        op1 = help(i+1, arr, selected, len)
        # skip
        for j in (0...currString.length)
            selected[currString[j].ord - "a".ord]=0
        end
        len -=currString.length
        op2 = help(i+1, arr, selected, len)
        return [op1,op2].max
    end
end
def compare(selected, currString)
    selfCheck = Array.new(26,0)
    for i in (0...currString.length)
        return false if selfCheck[currString[i].ord - "a".ord] == 1
        selfCheck[currString[i].ord - "a".ord] = 1
    end
    for i in (0...currString.length)
        return false if selected[currString[i].ord - "a".ord] == 1
    end 
    return true
end

# Pseudo-Palinfromic Paths in a Binary Tree
def pseudo_palindromic_paths (root)
    res = 0
    memo = Array.new(9, 0)

    job = -> node, memo do
        memo[node.val - 1] += 1
        left, right = !node.left.nil?, !node.right.nil?
        job.call(node.left, memo.dup) if left
        job.call(node.right, memo.dup) if right
        unless left || right    # We have reached a leaf node.
            res += 1 if memo.count(&:odd?) < 2
        end
    end

    job.call(root, memo)
    res

end

# Longest common sequence 
def longest_common_subsequence(text1, text2)
    dp = Array.new(text1.length + 1) { Array.new(text2.length + 1, 0) }
  
    (1..text1.length).each do |i|
      (1..text2.length).each do |j|
        dp[i][j] = text1[i - 1] == text2[j - 1] ? dp[i - 1][j - 1] + 1 : [dp[i - 1][j], dp[i][j - 1]].max
      end
    end
  
    dp[text1.length][text2.length]
  end

# Out of Boundary Paths
def find_paths(m, n, max_move, start_row, start_column)
    map = Array.new(m) { Array.new(n) { Array.new(max_move + 1) } }
    dp(m, n, max_move, start_row, start_column, map) % (10**9 + 7)
  end
  
  def dp(m, n, moves_left, i, j, map)
    return 1 if i < 0 || j < 0 || i >= m || j >= n
    return 0 if moves_left < 1
  
    map[i][j][moves_left] ||= begin
      paths = 0
  
      paths += dp(m, n, moves_left - 1, i + 1, j, map)
      paths += dp(m, n, moves_left - 1, i - 1, j, map)
      paths += dp(m, n, moves_left - 1, i, j + 1, map)
      paths + dp(m, n, moves_left - 1, i, j - 1, map)
    end
  end

  # Implement Queue using Stacks
  class MyQueue
    def initialize
      @queue = []
    end
  
    def push(x)
      @queue.push(x)
    end
  
    def pop
      @queue.shift
    end
  
    def peek
      @queue.first
    end
  
    def empty
      @queue.empty?
    end
  end


# Evaluate Reverse Polish Notation
OPERATIONS = %w[+ - / *].to_set

def eval_rpn(tokens)
  stack = []
  tokens.each do |item|
    next stack << item.to_i if !OPERATIONS.include?(item)

    val2, val1 = stack.pop, stack.pop
    stack << val1.to_f.send(item, val2).to_i # because 6/-132 = -1
  end

  stack[0]
end

# Daily Temperature
def daily_temperatures(temps)
    results = Array.new(temps.length, 0)
    stack = []
    # UPVOTE !
    temps.each_with_index do |temp, i|
      while !stack.empty? && temps[stack.last] < temp
        index = stack.pop
        results[index] = i - index
      end
      stack.push(i)
    end

    results
  end

# Sequential Digits
def queue_mega_generator size
    answer = []
    start = 0
    while start < 9
        digit_string = ""
        tmp = start 
        size.times do
            tmp = tmp + 1
            break if tmp > 9
            digit_string += tmp.to_s 
        end
        break if digit_string.size < size
        answer.push digit_string.to_i
        start += 1
    end
    answer
end
# @param {Integer} low
# @param {Integer} high
# @return {Integer[]}
def sequential_digits(low, high)
    answer = []
    min = low.to_s.size
    max = high.to_s.size
    (min..max).each do |size|
        answer += queue_mega_generator(size).select{|x| x.between?(low,high)}
    end
    answer
end

# Furst unique character
def first_uniq_char(s)
    freq = s.chars.tally 

    s.chars.each_with_index do |char, i|
        return i if freq[char] == 1 
    end 

    return -1 
end


# ZigZag conversion
def convert(s, num_rows)
    return s if num_rows == 1
    
    total_indices = (num_rows - 1) * 2
    result = Array.new(num_rows) { "" }
    
    s.each_char.with_index do |char, index|
      index %= total_indices
      index = total_indices - index if index >= num_rows
      result[index] << char
    end
  
    result.join
  end

# Sort by frequecies
def frequency_sort(s)
    result = ''
    arr = s.split('').sort!
    h = Hash.new {|h, k| h[k] = ''}
    p1 = p2 = 0
    while(p1 < arr.size || p2 < arr.size)
        if arr[p1] == arr[p2]
            p2 += 1
        else
            size = p2 - p1
            h[size] += arr[p1...p2].join('')
            p1 = p2
        end
    end
    h.keys.sort.reverse.each {|key| result += h[key] }
    result
end

# perfect squares
def num_squares(n)
    n /= 4 while n % 4 == 0
    return 4 if n % 8 == 7
    return 3 if n.prime_division.any? { |p, e| p % 4 == 3 && e.odd? }
    (n**0.5).to_i**2 == n ? 1 : 2
  end


#Largest divisible subset
def largest_divisible_subset(nums)
    @nums, @cache, max = nums.sort, {}, []
  
    @nums.each_with_index do |_, i|
      subset = dp(i)
      max = subset if subset.length > max.length
    end
  
    max
  end
  
  def dp(i)
    return @cache[i] if @cache.key?(i)
    
    max_subset = []
    (i+1...@nums.length).each do |j| # Start from i+1 to avoid self-comparison
      if @nums[j] % @nums[i] == 0
        subset = dp(j)
        max_subset = subset if subset.length > max_subset.length
      end
    end
  
    @cache[i] = [@nums[i]] + max_subset # Correct reference to @nums
  end

  # Majority Element
  def majority_element(nums)

    n = (nums.length / 2.0)
    counter = Hash.new(0)

    nums.each do |num|
        counter[num] += 1
        if counter[num] > n 
            return num
        end
    end
    

end


# Find First Palinfromic String in the Array
def first_palindrome(words)
    first = ""
    words.each do |word|
      first = word
      if word.eql?(first.reverse)
        return word
      end
    end
    return ""
  end


# Find Polygon with the largest perimeter
def largestPerimeter(nums)
    nums.sort!
    ans = -1
    n = nums.size
    prefix = Array.new(n, 0)
    prefix[0] = nums[0]
    for i in 1...n
        prefix[i] = prefix[i-1] + nums[i]
    end
    for i in 2...nums.size
        if prefix[i] - nums[i] > nums[i]
            ans = prefix[i]
        end
    end
    return ans
end


def furthest_building(heights, bricks, ladders)
    left = 0
    right = heights.count - 1
    result = 0
    
    while left <= right
      mid = (left + right)/2
      if can_reach?(heights[0...mid+1], bricks, ladders)
        result = mid
        left = mid + 1
      else
        right = mid - 1
      end
    end
    
    result
  end
  
  def can_reach?(heights, bricks, ladders)
    diffs = []
    n = heights.count
    
    heights.each_with_index do |h, i|
      break if i >= n - 1
      diffs << heights[i+1] - h if h < heights[i+1]
    end
    
    diffs.sort!
    
    diffs.each do |diff|
      if bricks >= diff
        bricks -= diff
      elsif ladders > 0
        ladders -= 1
      else
        return false
      end
    end
    
    true
  end

  # reverse integer
  def reverse(x)
    a = x.to_s
    if a.reverse.to_i > 2147483648
    return 0
    end
    if x < 0
    return ("-" + a.reverse.chop).to_i
    
    else
    return a.reverse.to_i
    end   
    end

# Power of 2
def is_power_of_two(n)
    return false if n == 0

    n & (n - 1) == 0
end

# Poison duration
def find_poisoned_duration(time_series, duration)
    if duration == 0
        return 0
    end
    

    last_t_poisoned = -1
    total = 0
    time_series.each do |t|
        if last_t_poisoned < t
            total += duration
            last_t_poisoned = t + duration - 1
        else
            next_last_t_poisoned = t + duration - 1
            total += next_last_t_poisoned - last_t_poisoned 
            last_t_poisoned = next_last_t_poisoned
        end

    end

    total
end


# Remove duplicates from an array 2
def remove_duplicates(nums)
    current = nums[0]
    count = i = 1
    while i < nums.size
        if nums[i] == current 
            count += 1 
            if count > 2 
                k = i+1 
                while nums[k] == nums[i]
                    k += 1
                end
                nums[i..] = nums[k..]
                count = 1
            end
        else 
            count = 1
        end
        current = nums[i]
        i += 1
    end
    
    nums.size
end

# first bad version
def first_bad_version(n)
    return n if !is_bad_version(n - 1) || n == 1
    return 1 if is_bad_version(1)
    left = 1
    right = n


    flag = false
    until flag
        middle = left + ((right - left) / 2)
        if is_bad_version(middle)
            right = middle
        else 
            left = middle
        end

        if left == right - 1
            if is_bad_version(left)
                return left
            end
            flag = true
        end
    end

    right
end

# Maximum Odd Binary Number
def maximum_odd_binary_number(s)
    ones = s.count('1')
    zeros = s.size - ones
  
    '1' * (ones - 1) + '0' * zeros + '1'
  end

  # remove nth element from the end
  def remove_nth_from_end(head, n)
    if head == nil or head.next == nil 
        return nil
    end
    dummy = ListNode.new(0)
    dummy.next = head
    count = 0

    iterate = head
    while iterate != nil
        count+=1
        iterate = iterate.next 
    end
    count-=n
    prev = dummy
    for a in 1..count do
        prev = prev.next
    end

    prev.next = prev.next.next
    return dummy.next
end

# minimum length of stirng after deleting similar ends

def minimum_length(s)
    l, r = 0, s.size - 1
    while l < r && s[l] == s[r] do
        c = s[l]
        l += 1 while l <= r && s[l] == c
        r -= 1 while r >= l && s[r] == c
    end
    r - l + 1    
end

# middle of a linked-list
def middle_node(head)
    fast = middle = head
    while fast && fast.next do
        middle = middle.next
        fast = (fast.next).next
    end
    return middle
end

# Reverse Integer
def reverse(x)
    s = x.abs.to_s.reverse
    return 0 if s.to_i > (2**31 - 1)
  
    x.negative? ? -s.to_i : s.to_i
end

#Minimum Common Value
def get_common(nums1, nums2)
    i = j = 0
  
    while i < nums1.count && j < nums2.count
      return nums1[i] if nums1[i] == nums2[j]
        
      nums1[i] > nums2[j] ? j += 1 : i += 1
    end
  
    -1
  end

# spiral matrix 2
def generate_matrix(n)
    # create the matrix and populate with nil
    matrix = Array.new(n) { |y| Array.new(n) { |x| nil } }

    # because 0 index, we decrease n by 1 to define the boundaries
    # for our matrix
    lim = n - 1

    # Short for direction x and direction y. Implies the velocity
    # against each axis. We use this to know how to modify the x,y
    # pointers during each iteration.
    dx, dy = 1, 0

    # x, y pointers initialized to row 0, col 0
    x, y = 0, 0

    # we iterate from 1 -> n*n (n*n is our target number)
    1.upto(n * n) do |num|
        # write the current num to the matrix at x, y
        matrix[y][x] = num

        # out of bounds!
        # basically this determines if the next iteration will put
        # the value of x or y at values less than 0 or greater than lim
        oob = x + dx > lim || x + dx < 0 || y + dy > lim || y + dy < 0
        
        # checks if the next position has already been written
        # to (i.e) is not nil. Use oob to actually
        # avoid reading out of bounds
        should_turn = oob || matrix[y + dy][x + dx] != nil

        # we basically want to turn right if the above condition is met
        if should_turn
            # so if the velocity of dy is 0
            # we want to follow the direction of dx
            if dy == 0
                dy = dx
                dx = 0

            # otherwise, it's implicit that dx is 0 if dy is not.
            # The inverse of the rule for dy applies here - if
            # dy was -1, dx should be 1. Seems weird, but go follow
            # the problem explanation, it checks out lol.
            else
                dx = -dy
                dy = 0
            end
        end

        # adjust the pointers using the velocity modifiers
        x += dx
        y += dy
    end

    matrix
end

# Remove Zero Sum Consecutive
def remove_zero_sum_sublists(head)
    nodes = []
    node = head
    total_index_hash = {}
    total = 0
    while node
        total += node.val 
        # Reset nodes and total_index_hash as our total
        # reached 0 meaning all nodes evened out.
        if total == 0
            nodes = []
            total_index_hash = {}
        else
            # If we ran into this total before...
            if total_index_hash[total]
                # Remove all node totals we have stored in our
                # total index hash based on the nodes
                # we are deleting from our nodes array.
                if nodes[(total_index_hash[total] + 1)..-1]
                    nodes[(total_index_hash[total] + 1)..-1].each do |node_arr|
                        total_index_hash.delete node_arr[1]
                    end
                end

                # Remove nodes from nodes array from the point
                # of where we hit our total in the past.
                nodes = nodes[0..total_index_hash[total]]
            # When we havent hit this total before we store it's index
            # and add our node to our nodes array.
            else

                total_index_hash[total] = nodes.size

                if node.val != 0
                    nodes << [node, total]
                end
            end
        end
        # Remove relationship between node and next node.
        prev_node = node
        
        # Set our node for our next iteration.
        node = node.next
        prev_node.next = nil
    end

    # Reassemble the linked list and return the head.
    if nodes.size > 0
        node = nodes[0][0]
        nodes[1..-1].each do |node_arr|
            node.next = node_arr[0]
            node = node_arr[0]
        end

        nodes[0][0]
    else
        nil
    end

    
end

# find the pivot integer
def pivot_integer(n)
    left_sum = (n * (n + 1)) / 2
    right_sum = 0

    while left_sum > right_sum
      right_sum += n
      return n if right_sum == left_sum
      left_sum -= n
      n -= 1
    end

    -1
  end


# Binary Subarrays with sum

def num_subarrays_with_sum(nums, goal)
    if goal == 0
      
  
      nums
        .chunk_while(&:==)  
        .filter{_1[0] == 0}
        .map(&:length)  
        .map{(_1 * _1 + _1) / 2} 
        .sum  
    else
       
  
      [1, *nums, 1]  
        .each_with_index 
        .filter{_1[0] == 1} 
        .map{_1[1]}
        .each_cons(goal + 2)  
        .map{_1.first(2) + _1.last(2)}  
        .map{|m, i, j, n| (i - m) * (n - j)} 
        .sum  
    end
  end


# Product of Array Except Self
def product_except_self(nums)
    answer = Array.new(nums.length, 1)

    nums.each_with_index do |num, index|
        (0...index).each do |index2|
            answer[index2] *= num
        end

        ((index + 1)...nums.length).each do |index3|
            answer[index3] *= num
        end

    end
    answer
end

# Rotate Image
def rotate(matrix)
    l = 0
    r = matrix.length - 1
  
    while l < r do 
      t = l
      b = r
      (0..r - l - 1).each do |i|
        # save top left
        top_left = matrix[t][l + i]
  
        # move bottom left to top left
        matrix[t][l + i] = matrix[b - i][l]
  
        # move bottom right to bottom left
        matrix[b - i][l] = matrix[b][r - i]
  
        # move top right to bottom right
        matrix[b][r - i] = matrix[t + i][r]
  
        # move top left to top right
        matrix[t + i][r] = top_left
      end
  
      l += 1
      r -= 1
    end
  
    matrix
  end

# Insert Interval
def insert(intervals, new_interval)
    result = []

    intervals.each_with_index do |interval, i|
        # no overlapping intervals
        if new_interval[1] < interval[0] # new_interval comes before first element in the intervals 
            result << new_interval
            return result + intervals[i...intervals.length]
        elsif new_interval[0] > interval[1] # new_interval comes after the current interval
            result << interval
        else # overlapping intervals
            new_interval = [[interval[0],new_interval[0]].min, [interval[1], new_interval[1]].max]
        end
    end
    result << new_interval
    result
end


# Remove Element
def remove_element(nums, val)

    counter = 0

    nums.each_with_index do |ele, index|
        
        if (ele == val)
            counter += 1
            nums[index] = 100000000
        end
    end


    nums.sort!
    
    return nums.length - counter
end

# Remove Element
def remove_element(nums, val)

    counter = 0

    nums.each_with_index do |ele, index|
        
        if (ele != val)
            nums[index] = 1000
            nums[counter] = ele
            counter += 1
        else 
            nums[index] = 1000
        end
    end


    return counter
end

#Duplicate Number
def find_duplicate(nums)
    h = nums.tally # constant space
    h.each { |k,v| return k if v > 1 }
    0
 end


# Find all duplicates in an array
def find_duplicates(nums)
    nums.map do |num|
      index = num.abs - 1

      if nums[index] < 0
        num.abs
      else
        nums[index] = -nums[index] and next
      end
    end.compact
end


# Reveal cards in increasing order
def deck_revealed_increasing(deck)
    sorted_deck = deck.sort

    final_deck = [sorted_deck.pop]

    until sorted_deck.length == 0
        top_card = sorted_deck.pop
        second_card = final_deck.pop

        final_deck = [top_card, second_card] + final_deck
    end

    return final_deck
end



# Trapping Rain Water
def find_max(array, side="L")
    max_value = array.max
    if side == "L"
      last_max_index = array.index(max_value)
    else
      last_max_index = array.rindex(max_value)
    end
    #puts "find_max: #{array} > #{side} > #{last_max_index}"
    last_max_index
  end
  
  def special_case(area)
    return (area.length < 3 || (area.length == 3 && (area[0]==0 || area[area.length-1] ==0)))
  end
  
  def calc_cap(area)
    #print "calc_cap area: #{area} " 
    
    if special_case(area)
      total = 0
    else
      r_wall = area.pop
      l_wall = area.shift 
      min_height = [l_wall, r_wall].min
      total = (min_height * area.length) - area.sum
    end
  #  puts "total:  #{total}"
    total 
  end
  
  def trap(height)
    # special cases
    return 0 if special_case(height)
    
    rango = height
    cap_total = 0
  
    r_rango = nil
    while rango.length > 2
      r_wall = find_max(rango, "R")
  
      # define right side if exists ;)     
      if r_rango.nil?
        r_rango = rango[r_wall..]
      end
  
      sub_rango = rango[0..r_wall-1]
      l_wall = find_max(sub_rango, "L")
      area = rango[l_wall..r_wall]
      cap_total += calc_cap(area)
  
      #define new rango
      rango = rango[0..l_wall]
    end
  
    # start with right side 
    rango = r_rango
    while rango.length > 2
      r_wall = find_max(rango[1..], "R")+1    
      sub_rango = rango[0..r_wall-1]
      l_wall = find_max(sub_rango, "L")
      area = rango[l_wall..r_wall]
      cap_total += calc_cap(area)
  
      #define new rango
      rango = rango[r_wall..rango.length-1]
     # puts "new rango: #{rango}" 
    end
  
    cap_total
  end


# Four Sum
def max_four(nums)
    count = nums.tally
    arr = []
    count.each do |k,v|
        ([v,4].min).times { arr << k }
    end
    arr
end


def four_sum(nums, target)
    nums = max_four(nums)
    res = Set[]
    
    #hash: sum is key, array of pairs of indices is value
    two_sum = Hash.new { |h,k| h[k] = [] }
    
    (0...nums.length).each do |a|
        (a+1...nums.length).each do |b|
            sum = nums[a] + nums[b]
            
            two_sum[target - sum].each do |pair|
                if ([a,b] + pair).uniq.length == 4
                    c,d = pair
                    res.add([nums[a],nums[b],nums[c],nums[d]].sort)
                end
            end
            
            two_sum[sum] << [a,b]
        end
    end
    
    res.to_a
end

# Maximal Rectangle
def maximal_rectangle(matrix)
    return 0 if matrix.nil? || matrix.length == 0 || matrix[0].length == 0
    
    m = matrix.length
    n = matrix[0].length
    
    heights = Array.new(n, 0)
    left_boundaries = Array.new(n, 0)
    right_boundaries = Array.new(n, n)
    
    max_rectangle = 0
    
    (0...m).each do |i|
        left = 0
        right = n
        
        update_heights_and_left_boundaries(matrix[i], heights, left_boundaries, left)
        
        update_right_boundaries(matrix[i], right_boundaries, right)
        
        max_rectangle = calculate_max_rectangle(heights, left_boundaries, right_boundaries, max_rectangle)
    end
    
    max_rectangle
end

def update_heights_and_left_boundaries(row, heights, left_boundaries, left)
    (0...heights.length).each do |j|
        if row[j] == '1'
            heights[j] += 1
            left_boundaries[j] = [left_boundaries[j], left].max
        else
            heights[j] = 0
            left_boundaries[j] = 0
            left = j + 1
        end
    end
end

def update_right_boundaries(row, right_boundaries, right)
    (right_boundaries.length - 1).downto(0) do |j|
        if row[j] == '1'
            right_boundaries[j] = [right_boundaries[j], right].min
        else
            right_boundaries[j] = right
            right = j
        end
    end
end

def calculate_max_rectangle(heights, left_boundaries, right_boundaries, max_rectangle)
    (0...heights.length).each do |j|
        width = right_boundaries[j] - left_boundaries[j]
        area = heights[j] * width
        max_rectangle = [max_rectangle, area].max
    end
    max_rectangle
end


# Sum root to leaf numbers
def sum_numbers(root)
    children(root, '').map(&:to_i).sum
end

def children(node, path)
  ans = []
  ans << children(node.left,  path + node.val.to_s) if node.left
  ans << children(node.right, path + node.val.to_s) if node.right
  ans << path + node.val.to_s if !node.left and !node.right
  ans.flatten
end


# Combination Sum
def combination_sum(candidates, target)
    candidates.sort.flat_map do |candidate|
     if candidate == target
         [[candidate]]
     elsif candidate < target
         combination_sum(candidates.select { |c| c >= candidate }, target - candidate)
             .map { |combo| [candidate] + combo }
     else
         []
     end
 end
end

# Merger alternatively
def merge_alternately(word1, word2)
    # while loop, where theres a counter and it keeps going until the length of one of the words

    # Adds the word index to the final word string 

    final_word = ""

    counter = 0

    word1_length = word1.length
    word2_length = word2.length 

    while counter != word1.length && counter != word2.length 
        final_word += word1[counter] + word2[counter]
        counter += 1
    end

    if counter == word1.length 
        return final_word + word2[counter..-1]
    else 
        return final_word + word1[counter..-1]
    end

    

end

# Find all groups of farmland
def find_farmland(land)
    farm_lends = []
    land.each_with_index do |row, i|
      row.each_with_index do |value, j|
        if value == FARMLEND && is_started_farmlend?(land, i, j)
          start_farm_lend = [i, j] 
          end_farm_lend = find_end_of_farmlend(land, i, j)
          farm_lends << start_farm_lend.concat(end_farm_lend) 
        end  
      end
    end
    farm_lends
  end
  
  def is_started_farmlend?(land, i, j)
    left_value = j == 0 ? FOREST : land.dig(i)&.dig(j - 1).to_i
    up_value = i == 0 ? FOREST : land.dig(i - 1)&.dig(j).to_i
    return true if (left_value + up_value).zero?
  
    false
  end
  
  def find_end_of_farmlend(land, i, j)
    return i, j  if is_ended_farmlend?(land, i, j)
    return find_end_of_farmlend(land, i, j + 1) if land.dig(i)&.dig(j + 1) == FARMLEND
    return find_end_of_farmlend(land, i + 1, j) if land.dig(i + 1)&.dig(j) == FARMLEND
    return find_end_of_farmlend(land, i + 1, j + 1)
  end
  
  def is_ended_farmlend?(land, i, j)
    right_value = land.dig(i)&.dig(j + 1).to_i
    bottom_value = land.dig(i + 1)&.dig(j).to_i
    return true if (right_value + bottom_value).zero?
  
    false
  end

  # Pow (X, Y)
  def my_pow(x, n)
    return 1 if n == 0

    return x if n == 1

    result = cal_power(x,n.abs)

    n < 0 ? (1/result) : result
end

def cal_power(x, n)
    return 1 if n == 0

    return x if n == 1

    result = cal_power(x, n/2)

    if n % 2 != 0 # n.odd? # check for odd
        result * result * x
    else
        result * result
    end
end 

# Product of Array except self
def product_except_self(nums)

    final_array = Array.new(nums.length)

    current_product = product_of_array(nums[1..-1]).to_f

    if current_product == 0 || nums[0] == 0
        final_array = product_if_zero(nums, final_array, current_product)
    else 
        final_array = product_if_norm(nums, final_array, current_product)
    end
    
    

    return final_array

end

def product_if_zero(array, final_array, current_product)
    product = array[0]

    if array[0] == 0
        product = 1
        array[1..-1].each_with_index do |num, index|
            product = product * num
            final_array[index + 1] = 0
        end

        final_array[0] = product
    else
        zero_index = 0 
        final_array[0] = 0
        array[1..-1].each_with_index do |num, index|

            if zero_index != 0 && num == 0 
                final_array[index + 1] = 0
                product = 0
            elsif num == 0
                zero_index = index + 1
            else 
                product = product * num
                final_array[index + 1] = 0            
            end

        end
        final_array[zero_index] = product

    end

    final_array
end

def product_if_norm(array, final_array, current_product)

    array.each_with_index do |num, index|
        if index == 0
            final_array[0] = current_product.to_i
        else          
            current_product = current_product / num
            current_product = current_product * array[index - 1]
            final_array[index] = current_product.to_i 
        end
    end

    final_array 
end

def product_of_array(array)
    product = 1
    array.each do |num|
        product = product * num
    end

    product
end


# Move zeroes
def move_zeroes(nums)
    counter = 0 

    zero_counter = nums.length - 1

    tracker = {}
 
    nums.each do |num|
        if num == 0
            tracker[zero_counter] = 0
            zero_counter -= 1
        else 
            tracker[counter] = num        
            counter += 1
        end
    end
    
    (0...nums.length).each do |count|
        nums[count] = tracker[count]
    end
    
    return nums
end


# Tribonacci Number
def tribonacci(n)
    trib_hash = { 0 => 0, 1 => 1, 2 => 1}

    counter = 3

    return trib_hash[n] if trib_hash[n]

    until trib_hash[n]
        trib_hash[counter] = trib_hash[counter - 1] + trib_hash[counter - 2] + trib_hash[counter - 3]
        counter += 1
    end

    trib_hash[n]
        

    
end


# Minimum falling path
def min_falling_path_sum(grid)
    @grid = grid
    @n = grid.length
    (0...@n).each do |i|
        (0...@n).each do |j|
            grid[i][j] = [grid[i][j],j]
        end
    end

    grid.map! { |row| row.sort }
    grid.map! { |row| row.length > 2 ? row[0..2] : row }

    @memo = {}

    min_sum(0,nil)
end

def min_sum(row,col)
    return 0 if row == @n
    return @memo[[row,col]] if @memo[[row,col]]

    options = []
    @grid[row].each do |arr|
        val,c = arr
        options << min_sum(row+1,c) + val unless col == c
    end

    @memo[[row,col]] = options.min
end

# freedom trail
def find_rotate_steps r, t
    return t.size if (r = r.bytes).uniq.size < 2
    s, z, r = {0 => 0}, r.size, r.each_index.group_by { r[_1] }
    t.bytes.each do
        e, cj = Hash.new(1e5), r[_1]
        s.each do | i, v |
            v += 1
            for j in cj
                d = (i - j).abs
                e[j] = [e[j], v + [d, z - d].min].min
            end
        end
        s = e
    end
    s.each_value.min
end

# Reverse Integer
def reverse(x)
    s = x.abs.to_s.reverse
    return 0 if s.to_i > (2**31 - 1)
  
    x.negative? ? -s.to_i : s.to_i
  end


# NExt Permutations
def next_permutation(nums)
    i = nums.length - 2
    while i >= 0 && nums[i] >= nums[i+1]
        i -= 1
    end
    if i >= 0
        j = nums.length - 1
        while j >=0 && nums[j] <= nums[i]
            j -= 1
        end
        nums[i], nums[j] = nums[j], nums[i]
    end
    nums[i + 1..-1] = nums[i + 1..-1].reverse
end

# Reverse Prefix of Word
def reverse_prefix(word, ch)
    prefix = ""

    word.each_char.with_index do |char, index|
        if char != ch
            prefix = char + prefix
        else 
            return char + prefix + word[(index + 1)..-1]
        end
    end
end

# Reverse Prefix of Word
def reverse_prefix(word, ch)
    prefix = ""

    index = 0

    while index < word.length
        if word[index] != ch
            prefix = word[index] + prefix
            index += 1
        else 
            return word[index] + prefix + word[(index + 1)..-1]
        end
    end

    return word
end

# Boats to save people
def num_rescue_boats(people, limit)
    people.sort!
    ships = 0
    left, right = 0, people.length - 1
    while left <= right
        if people[left] + people[right] <= limit
            left += 1
        end
        right -= 1
        ships += 1
    end
    ships
end


# Delete Node
def delete_node(node)
    node.val = node.next.val
    node.next = node.next.next
end

# Combination Sum 
def combination_sum(candidates, target)
    candidates.sort.flat_map do |candidate|
     if candidate == target
         [[candidate]]
     elsif candidate < target
         combination_sum(candidates.select { |c| c >= candidate }, target - candidate)
             .map { |combo| [candidate] + combo }
     else
         []
     end
 end
end


# Remove Nodes from Linked List
def remove_nodes(head)

    # Reverse the linked list

    # Then compare, changing the comparer if a greater value comes up

    return head if !head.next

    head = reverse_list(head)

    comparer = head.val
    
    current_node = head.next

    previous_node = head

    while current_node 
        if current_node.val < comparer
            previous_node.next = current_node.next
            current_node = current_node.next
        else 
            comparer = current_node.val
            previous_node = current_node
            current_node = current_node.next
        end
    end

    return reverse_list(head)
end

def reverse_list(head)

    return head if !head.next

    current_node = head.next 

    previous_node = head 

    previous_node.next = nil

    while true
        next_node = current_node.next 

        current_node.next = previous_node 

        previous_node = current_node

        if next_node
            current_node = next_node
        else 
            return current_node
        end
    end

end

# Double a Linked List
def double_it(head)

    # Record the complete number as a string
    number = ""

    current_node = head 

    while current_node
        number += current_node.val.to_s
        current_node = current_node.next
    end

    # Double the number and split the digits into an array
    number = number.to_i * 2

    array = number.to_s.split("").map{ |num| num.to_i }
    
    # Create a new Linked List, which uses the array to create the nodes
    new_head = ListNode.new(array.shift())

    current_node = new_head

    array.each do |num|
        current_node.next = ListNode.new(num)
        current_node = current_node.next
    end

    return new_head
end

# Relative Ranks
def find_relative_ranks(score)
    ordered = score.sort().reverse()
    
    placement = 3

    ranks = {
        0 => "Gold Medal",
        1 => "Silver Medal",
        2 => "Bronze Medal"
    }

    placements = {}

    ordered.each_with_index do |num, index|
        if ranks[index]
            placements[num] = ranks[index]
        else 
            placement += 1
            placements[num] = placement.to_s
        end
    end

    return_array = []

    score.each do |num|
        return_array.push(placements[num])
    end

    return_array
end

# Maximum Happiness
def maximum_happiness_sum(happiness, k)
    sorted = happiness.sort

    happiness_change = 0

    total_happiness = 0
    
    until k == 0
        most_happy = sorted.pop

        most_happy -= happiness_change

        total_happiness += most_happy if most_happy > 0

        happiness_change += 1
        
        k -= 1

    end

    total_happiness


    
end


# Minimum Cost to Hire K Workers
def initialize array = [], heapify = true, &is_unordered
    raise ArgumentError.new 'PQ init' unless
        array.class == Array &&
        (heapify == true || heapify == false) &&
        block_given?
    @a, @u = array, is_unordered
    return unless heapify
    i = @a.size / 2
    sink i while (i -= 1) >= 0
end

def size = @a.size
def empty? = @a.empty?
def top = @a.first
def push_pop(x) = !@a.empty? && @u.(x, @a.first) ? pop_push(x) : x

def pop_push x
    t, @a[0], = @a.first, x
    sink 0
    t
end

def << x
    i = @a.size
    @a << x
    while i > 0
        p = (i - 1) / 2
        break unless @u.call @a[p], @a[i]
        @a[p], @a[i] = @a[i], @a[p]
        i = p
    end
    self
end

def pop
    return @a.pop if @a.size < 2
    t, @a[0] = @a.first, @a.pop
    sink 0
    t
end

private

def sink p
    z = @a.size
    while (c = p * 2 + 1) < z
        r = c + 1
        c = r if r < z && @u.(@a[c], @a[r])
        break unless @u.call @a[p], @a[c]
        @a[p], @a[c] = @a[c], @a[p]
        p = c
    end
end

end

def mincost_to_hire_workers e, w, k
n = e.size
n.times { w[_1] = w[_1] .fdiv e[_1] }
a = n.times.sort_by { w[_1] }
f, a = a[0, k], a[k..]
s = f.sum { e[_1] }
m = s * w[f.last]
q = PQ.new(f) { e[_1] < e[_2] }
for i in a
    s += e[i]
    s -= e[q.pop_push i]
    x = s * w[i]
    m = x if m > x
end
m
end


# 3 Sum closest
def three_sum_closest(nums, target)
    nums.sort!
    # Set default ans so we don't have to be concerned with nil.
    ans = 9999999999
    ans_abs = 9999999999
    # j and k must be bigger than i.
    i_index_stop = nums.size - 2
    i = 0
    while i < i_index_stop
        # Skip repeating numbers. 
        if !(i > 0 and nums[i] == nums[i-1])
            j = i + 1
            k = nums.size - 1
            while j < k
                total = nums[i] + nums[j] + nums[k]
                if total == target
                    return target
                elsif total > target
                    k -= 1
                else
                    j += 1
                end

                total_abs = (total - target).abs
                
                
                if total_abs < ans_abs
                    ans = total
                    ans_abs = total_abs
                end
            end
        end

        i += 1
    end

    ans
end

# Reverse Integer
def reverse(x)

    upper_limit = (2 ** 31) - 1
    lower_limit = -2 ** 31

    return 0 if x > upper_limit || x < lower_limit

    flag = x > 0 ? true : false
    stringified = x.to_s

    reversed = stringified.reverse

    reversed_integer = reversed.to_i
    
    return 0 if reversed_integer > upper_limit || reversed_integer < lower_limit

    if flag
        return reversed_integer
    else 
        return 0 - reversed_integer
    end
end

# Counting triplets
def count_triplets(arr)
    result = 0
    l = arr.size
    (0..l - 2).each do |m|
      a = nil
      (m..l - 2).each do |i|
        a.nil? ? a = arr[i] : a ^= arr[i]
        b = nil
        (i + 1..l - 1).each do |k|
          b.nil? ? b = arr[k] : b ^= arr[k]
          result += 1 if a ==  b
        end
      end
    end
    result
  end


# Single Number 3
def single_number(nums)
    a = nums.reduce(:^)
    b = nums.select { |n| n & a & -a > 0 }.reduce(:^)
    [a ^ b, b]
  end


# Score of string
def score_of_string(s)
    res = 0
        for i in 0...s.length - 1
            res += (s[i].ord - s[i + 1].ord).abs
        end
    res
end

#  Append Characters to String to Make Subsequence
def append_characters(s, t)
    tlength = t.length
    return tlength if s.length == 0

    index = 0
    s.chars.each_with_index do |c, i|
        if index < tlength && t[index] == c
            index += 1
        end
    end
    tlength - index
end

# Longest Palindrome
str_count = Array.new(52, 0)
    
s.each_char do |char|
  if char >= 'a' && char <= 'z'
    str_count[char.ord - 'a'.ord + 26] += 1
  else
    str_count[char.ord - 'A'.ord] += 1
  end
end

ans = 0
has_odd = false

str_count.each do |count|
  if count % 2 == 0
    ans += count
  else
    ans += count - 1
    has_odd = true
  end
end

ans += 1 if has_odd  
ans
end

# Common Chars
def common_chars(words)
    min_freq = Array.new(26, Float::INFINITY)
    
    words.each do |word|
      freq = Array.new(26, 0)
      word.each_char do |char|
        freq[char.ord - 'a'.ord] += 1
      end
      (0...26).each do |i|
        min_freq[i] = [min_freq[i], freq[i]].min
      end
    end
    
    result = []
    (0...26).each do |i|
      while min_freq[i] > 0
        result << (i + 'a'.ord).chr
        min_freq[i] -= 1
      end
    end
    
    result
end

# is straight hand
def is_n_straight_hand(hand, group_size)
    if (hand.size % group_size != 0)
        return false
    end

    hand.sort!
    while hand.size > 0
        count = 1
        last_num = hand.shift
        while count < group_size
            next_num = last_num + 1
            index = hand.bsearch_index{|n| next_num <=> n}
            if !index
                return false
            else
                hand.delete_at(index)
            end
            last_num = next_num
            count = count + 1
        end
    end

    return true
end


# Continuous subarray sums
def check_subarray_sum(nums, k)
    hash_map = { 0 => 0 }
    sum = 0

    nums.each_with_index do |num, i|
        sum += num

        # If the remainder sum % k occurs for the first time
        if !hash_map.key?(sum % k)
            hash_map[sum % k] = i + 1
        # If the subarray size is at least two
        elsif hash_map[sum % k] < i
            return true
        end
    end

    false
end

# String Compression
def compress(chars)
    walker, runner = 0, 0

    while runner < chars.size
        chars[walker] = chars[runner]
        count = 1
        while chars[runner] == chars[runner + 1]
            runner += 1
            count += 1
        end
        if count > 1
            count.to_s.each_char do |c|
                chars[walker += 1] = c
            end
        end
        runner += 1
        walker += 1
    end
    
    walker
end

# Relative Sort Array
require 'set'

def relative_sort_array(arr1, arr2)
    set, ord = Set[*arr2], arr2.each_with_index.each_with_object([]) {|(v, i), res| res[v] = i }
    aa = arr1.partition {|v| set.include?(v) }
    aa.first.sort! {|v1, v2| ord[v1] <=> ord[v2] } + aa.last.sort!
end

# Sorting
def sort_colors(nums)
    c0, c1 = nums.count(0), nums.count(1)  
    nums.fill(0, 0, c0).fill(1, c0, c1).fill(2, c0 + c1)
  end


# IPO
def bs_insert(a, item) =
    a.insert(a.bsearch_index { item <= _1 } || a.size, item)
  
  def find_maximized_capital(k, w, profits, costs)
    tasks = costs.zip(profits).sort
    sorted_profits = []
  
    k.times.reduce(w) do |capital, _|
      while tasks[0] && capital >= tasks[0][0]
        bs_insert(sorted_profits, tasks.shift[1])
      end
  
      capital + (sorted_profits.pop || 0)
    end
  end

# Valid Sodoku
def is_valid_sudoku(board)
    [
        has_valid_rows?(board),
        has_valid_cols?(board),
        has_valid_grids?(board)
    ].all?
end

def has_valid_rows?(board)
    board.all? { |row| _is_valid_array?(row) }
end

def has_valid_cols?(board)
    (0...9).each do |n|
        col = []
        board.each { |row| col << row[n] }
        return false unless _is_valid_array?(col)
    end

    true
end

def has_valid_grids?(board)
    [0,3,6].each do |n|
        [0,3,6].each do |i|
            grid = [
                board[n][i], board[n][i+1], board[n][i+2],
                board[n+1][i], board[n+1][i+1], board[n+1][i+2],
                board[n+2][i], board[n+2][i+1], board[n+2][i+2]
            ]

            return false unless _is_valid_array?(grid)
        end
    end

    true
end

def _is_valid_array?(arr)
    ref = [0] * 10

    arr.each do |sq|
        next if sq == "."
        return false if sq.to_i.zero? || sq.to_i > 9 || ref[sq.to_i] == 1
        ref[sq.to_i] = 1
    end

    true
end 
# Rotate image
def rotate(matrix)
    l = 0
    r = matrix.length - 1
  
    while l < r do 
      t = l
      b = r
      (0..r - l - 1).each do |i|
        # save top left
        top_left = matrix[t][l + i]
  
        # move bottom left to top left
        matrix[t][l + i] = matrix[b - i][l]
  
        # move bottom right to bottom left
        matrix[b - i][l] = matrix[b][r - i]
  
        # move top right to bottom right
        matrix[b][r - i] = matrix[t + i][r]
  
        # move top left to top right
        matrix[t + i][r] = top_left
      end
  
      l += 1
      r -= 1
    end
  
    matrix
  end

# Max Profit
def max_profit_assignment(difficulty, profit, worker)
    # Sort workers
    worker.sort_by!{|a| -a}
    max_diff = worker[0]

    # Create diff profit array where we pair difficulty with profit.
    diff_profit = []
    i = 0
    while i < difficulty.size
        if max_diff >= difficulty[i]
            diff_profit << [difficulty[i], profit[i]]
        end
        i += 1
    end

    # Sort by highest profit.
    diff_profit.sort_by!{|a| -a[1]}

    # n^2 time complexity
    # We reduce this by using a pointer to account for when a worker
    # runs into a job which is of higher difficulty than what the worker 
    # can profit. When this happens we increase a pointer(diff_profit_i)
    # so when the next worker is assigned a job, it starts from the same position
    # in diff_profit as the last worker. This removes the traversal of going through jobs
    # which are too hard for the worker and saves a ton of time.
    profit = 0
    diff_profit_i = 0
    worker.each do |worker_diff|
        diff_profit[diff_profit_i..-1].each do |job|
            if job[0] <= worker_diff
                profit += job[1]
                break
            else
                diff_profit_i += 1
            end 
        end
    end

    profit
end

# MAgnetic Force between Balls
def max_distance(position, m)
    position.sort!
  
    can_fit = ->(target) do
      count = 1
  
      current = position[0]
      (1...position.length).each do |i|
        if position[i] - current >= target
          current = position[i]
          count += 1
        end
      end
  
      count
    end
  
    left = 1
    right = position.last - position.first
    while left <= right
      middle = (left + right) / 2
  
      if can_fit.call(middle) >= m
        left = middle + 1
      else
        right = middle - 1
      end
    end
  
    right
  end


# Number of Nice Subarrays
def numberOfSubarrays_v1(nums, k)
    count = 0
    n = nums.length
    ans = 0
    i = 0
    for j in 0...n
        if nums[j] % 2 == 1
            count += 1
        end
        while count > k
            if nums[i] % 2 == 1
                count -= 1
            end
            i += 1
        end
        p = i
        count1 = count
        while count1 == k
            ans += 1
            if nums[p] % 2 == 1
                count1 -= 1
            end
            p += 1
        end
    end
    return ans
end

# Simplify Path
def simplify_path(path)
    arr = []
    path.split('/').each do |ele|
        next if ele.empty?
        if ele == ".."
            arr.pop() if arr.any?
        elsif ele == "."
            next
        else
            arr.push(ele)
        end
    end
    "/" + arr.join('/')
end


# Minimum arrow shots
def find_min_arrow_shots(points)
    points.sort_by!(&:last)
    points.each_with_object([1, points.first.last]) {|(pb, pe), res|
        if pb > res.last then
            res[0] += 1
            res[1] = pe
        end
    }.first
end

# Binary tree to greater sum tree
def bst_to_gst(root)
    @sum = 0
    def reverse_order_traversal(root)
        return nil if(root == nil)
        
        reverse_order_traversal(root.right)
        @sum += root.val
        root.val = @sum
        reverse_order_traversal(root.left)
        return root
    end
    reverse_order_traversal(root)
end

#Balance of a Binary Search Tree
def to_array(root)
    return if root.nil?
  
    to_array(root.left)
    @array << root
    to_array(root.right)
  end
  
  def balance_tree(array)
    return nil if array.empty?
  
    mid = array.size / 2
    root = array[mid]
    root.left = balance_tree(array[0...mid])
    root.right = balance_tree(array[mid+1..])
  
    root
  end
  
  def balance_bst(root)
    @array = []
    to_array(root)
    balance_tree(@array)
  end


# Center of star Graph
def find_center(edges)
    adjacency = Hash.new { |h, k| h[k] = [] }
  
    edges.each do |a, b|
      adjacency[a] << b
      adjacency[b] << a
    end
  
    adjacency.find { |k, v| (adjacency.keys - [k]).difference(v).none? }.first    
  end

# Maximum Total Importance of Roads
def maximum_importance(n, roads)
    l = Array.new(n, 0)
    roads.each do |road|
        road.each do |city|
            l[city] += 1
        end
    end
    l.sort!.reverse!
    ans = 0
    j = 0
    (n).downto(1) do |i|
        if j < n
            ans += l[j] * i
            j += 1
        end
    end
    return ans
end

# Ancestors of a Node in a DIrected Acyclic Graph
def get_ancestors(n, edges)
    ancestors = Array.new(n) {Set.new}
    children = Array.new(n) {[]}
    in_degree = Array.new(n, 0)
  
    edges.each {|from, to|
      in_degree[to] += 1
      children[from] << to
    }
  
    queue = []
    in_degree.each_with_index{|degree, node|
      queue << node if degree.zero?
    }
  
    order_of_nodes = []
    until queue.empty?
      node = queue.shift
      order_of_nodes << node
  
      children[node].each {|child|
        in_degree[child] -= 1
        queue << child if in_degree[child].zero?
      }
    end
  
    order_of_nodes.each {|node|
      children[node].each {|child|
        ancestors[child] += [node] + ancestors[node].to_a
      }
    }
  
    ancestors.map{_1.to_a.sort}
  end

# Largest Perimeter Triangle
def largestPerimeter(self, nums: List[int]) -> int:
    nums = sorted(nums)[::-1]
    for i in range(len(nums) - 2):
        if nums[i] < nums[i + 1] + nums[i + 2]:
            return nums[i] + nums[i + 1] + nums[i + 2]
    return 0

# Minimum Difference Between Largest and Smallest Value in Three Moves 
def minDifference(self, nums: List[int]) -> int:
    if len(nums) <= 4:
        return 0
    nums.sort()
    ans = nums[-1] - nums[0]
    for i in range(4):
        ans = min(ans, nums[-(4 - i)] - nums[i])
    return ans

# Merge Nodes in Between Zeros
def merge_nodes(head)
    f = head
    s = head
  
    loop do
      f = f.next
      if f.val == 0
        if f.next.nil?
          s.next = nil
          return head
        end
        s = s.next = f
      else
        s.val += f.val
      end
    end
  end

# Average Wait Time
def average_waiting_time(customers)
    next_cook_time = 0
    total_wait_time = 0

    customers.each do | customer |
        arrival_time = customer[0]
        cook_time = customer[1]

        if next_cook_time <= arrival_time
            total_wait_time = total_wait_time + (cook_time)
            next_cook_time = arrival_time + cook_time

        elsif next_cook_time > arrival_time
            # When did they arrive and how far away from the wait time is it? Subtract this number from the cook time and average
            total_wait_time = total_wait_time + ((next_cook_time - arrival_time) + cook_time)
            next_cook_time = next_cook_time + cook_time
        end
    end
    (total_wait_time.to_f / customers.length).to_f
end


# Crawler Log Folder
def min_operations(logs)
    stack = []
  
    logs.each do |command|
      if command == '../'
        stack.pop
      elsif command != './'
        stack.push(command)
      end
    end
  
    stack.length
  end

# Reverse Parenthesis
def reverse_parentheses(s)
    stack = []
    word = ''
  
    s.each_char {|c|
      if c == '('
        stack << word
        word = ''
      elsif c == ')'
        word = stack.pop + word.reverse
      else
        word += c
      end
    }
  
    word
  end

# Directions from a Binary Tree Node to Another
def dfs(start, target)
    path = ""
    stack = [[start, path]]
    while(!stack.empty?)
        cur_node, cur_path = stack.pop
        return cur_path if cur_node.val == target
        stack.push([cur_node.left, cur_path + 'L']) if cur_node && cur_node.left
        stack.push([cur_node.right, cur_path + 'R']) if cur_node && cur_node.right
    end
    return path
end


def get_directions(root, start_value, dest_value)
    start_path = dfs(root, start_value)
    dest_path = dfs(root, dest_value)
    i = 0
    while(start_path[i] == dest_path[i])
        i += 1
    end
    "U"*(start_path.length-i) + dest_path[i..-1]
end

# Delete Nodes and Return Forest
class TreeNode
    attr_accessor :parent
  end
  
  def del_nodes(root, to_delete)
    find_node_and_resolve_parent = lambda do |current, target_val|
      queue = [current]
      until queue.empty?
        current = queue.shift
  
        return current if current.val == target_val
  
        if current.left
          current.left.parent = current
          queue.append(current.left)
        end
        if current.right
          current.right.parent = current
          queue.append(current.right)
        end
      end
    end
  
    dummy = TreeNode.new(nil, root)
    nodes_to_delete = to_delete.map { find_node_and_resolve_parent.call(dummy, _1) }
  
    results = []
    nodes_to_delete.each do |node|
      results << node.left if node.left && !nodes_to_delete.include?(node.left)
      results << node.right if node.right && !nodes_to_delete.include?(node.right)
  
      node.parent.left = nil if node.parent.left == node
      node.parent.right = nil if node.parent.right == node
    end
  
    ([dummy.left] + results).compact
  end

# Number of good leaf nodes pairs
def count_pairs(root, distance)
    ans = 0
    dfs = lambda do |node|
      return [] unless node
      return [1] if node.left.nil? && node.right.nil?
  
      left_distance = dfs.call(node.left)
      right_distance = dfs.call(node.right)
  
      left_distance.each do |l|
        right_distance.each do |r|
          ans += 1 if l + r <= distance
        end
      end
  
      (left_distance + right_distance).map { |d| d + 1 }
    end
    dfs.call(root)
    ans
  end

# Sort array by increasing frequency 
def frequency_sort(nums)
    # Step 1: Create a frequency hash to store the count of each number
    freq = Hash.new(0)
    nums.each { |num| freq[num] += 1 }

    # Step 2: Create a hash to store numbers grouped by their frequency
    arrange = Hash.new { |hash, key| hash[key] = [] }

    # Step 3: Populate the arrange hash
    freq.each do |num, count|
        arrange[count] << num
    end

    # Step 4: Clear the nums array to prepare for the sorted result
    nums.clear

    # Step 5: Rebuild the nums array based on the frequencies
    arrange.keys.sort.each do |count|
        arrange[count].sort.reverse.each do |num|
            count.times { nums << num }
        end
    end

    nums
end


# Sort array using quick sort
def sort_array(nums)
    return nums if nums.length < 2
    @nums = nums
    quick_sort(0, @nums.size - 1)
end

def quick_sort(left, right)
    return if left >= right
    index = partition(left, right)
    
    quick_sort(left, index - 1)
    quick_sort(index, right)
    
    @nums
end

def partition(l, r)
    pivot = @nums[(l+r)/2]
    while l <= r
        l += 1 while @nums[l] < pivot
        r -= 1 while @nums[r] > pivot
            
        if l <= r
            swap(l, r)
            l += 1
            r -= 1
        end
    end
    l 
end
        
def swap(l, r)
    @nums[l], @nums[r] = @nums[r], @nums[l]
end

# Count and Say
def count_and_say(n)
    return "1" if n == 1  # Base case
    ans, temp = "", "1"
    
    for i in 2..n
      cnt = 0
      prev = temp[0]
      
      temp.chars.each_with_index do |char, j|
        if prev == char
          cnt += 1  # brute force
        else
          ans += cnt.to_s  # adding count to the end
          ans += temp[j-1]
          prev = char
          cnt = 1
        end
      end
      
      ans += cnt.to_s
      ans += temp[-1]
      temp = ans  # assigning ans to temp and making ans = ""
      ans = ""  # so that we can iterate over temp again!
    end
    
    temp
  end

  # Minimum swaps to group all 1's together
  def min_swaps(nums)
    count_ones = nums.count(1)
    count_zeros = nums[0...count_ones].count(0)
    ans = count_zeros
    arr = nums + nums[0...count_ones]
    (1...arr.length - count_ones).each do |i|
      count_zeros -= 1 if arr[i - 1] == 0
      count_zeros += 1 if arr[i + count_ones - 1] == 0
      ans = [ans, count_zeros].min
    end
    ans
  end

  # Single Number 3
  def single_number(nums)
    hash = Hash.new(0)

    nums.each do |num|
        hash[num] += 1
    end
    hash.select{ |key, value| value == 1 }.keys
end

# Nuumber to words

LESS_THAN_20 = [
    "", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine",
    "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen",
    "Seventeen", "Eighteen", "Nineteen"
  ]
  TENS = [
    "", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"
  ]
  THOUSANDS = ["", "Thousand", "Million", "Billion"]

  def number_to_words(num)
    return "Zero" if num == 0

    i = 0
    words = ""

    while num > 0
      if num % 1000 != 0
        words = helper(num % 1000) + THOUSANDS[i] + " " + words
      end
      num /= 1000
      i += 1
    end

    words.strip
  end

  private

  def helper(num)
    if num == 0
      ""
    elsif num < 20
      LESS_THAN_20[num] + " "
    elsif num < 100
      TENS[num / 10] + " " + helper(num % 10)
    else
      LESS_THAN_20[num / 100] + " Hundred " + helper(num % 100)
    end
  end
end

# Spiral Matrix 3
def spiral_matrix_iii(rows, cols, r_start, c_start)
    result = []
    directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
    dir_index = 0
    step_count = 1
    r, c = r_start, c_start
    result << [r, c]
    
    while result.size < rows * cols
        step_count.times do
            r += directions[dir_index][0]
            c += directions[dir_index][1]
            result << [r, c] if r >= 0 && r < rows && c >= 0 && c < cols
        end
        dir_index = (dir_index + 1) % 4
        step_count += 1 if dir_index % 2 == 0
    end
    
    result
end


# Magic Square
def num_magic_squares_inside(grid)
    def is_magic_square(grid, row, col)
        values = []
        seen = Array.new(10, false)

        (0...3).each do |i|
            (0...3).each do |j|
                val = grid[row + i][col + j]
                return false if val < 1 || val > 9 || seen[val]
                seen[val] = true
                values.push(val)
            end
        end

        sum = values[0] + values[1] + values[2]
        (0...3).each do |i|
            return false if values[i * 3] + values[i * 3 + 1] + values[i * 3 + 2] != sum
            return false if values[i] + values[i + 3] + values[i + 6] != sum
        end

        return false if values[0] + values[4] + values[8] != sum
        return false if values[2] + values[4] + values[6] != sum

        true
    end

    count = 0
    rows = grid.length
    cols = grid[0].length

    (0..rows - 3).each do |i|
        (0..cols - 3).each do |j|
            count += 1 if is_magic_square(grid, i, j)
        end
    end

    count
end

# Smallest pair distance
def smallest_distance_pair(nums, k)
    nums.sort!
    n = nums.size
  
    count_pairs = lambda do |x|
      cnt = 0
      r = 1
      n.times do |l|
        r += 1 while r < n && nums[r] - nums[l] <= x
        cnt += r - l - 1
      end
      cnt
    end
  
    ok = nums[-1] - nums[0]
    ng = -1
    while ok - ng > 1
      mid = (ok + ng) / 2
      count = count_pairs.call(mid)
      if count >= k
        ok = mid
      else
        ng = mid
      end
    end
    ok
  end

# 2 keys beyboard
def min_steps(n)
    steps = 0
    i = 2
    while i <= n
      while n % i == 0
        steps += i
        n /= i
      end
      i += 1
    end
    steps
  end


# 2 stone game
def stone_game_ii(piles)
    @memo = {}
    @piles = piles
    score = score(0,1)
    sum = piles.sum - score
    score + sum / 2
end

def score(idx,max)
    return 0 if idx >= @piles.length
    return @memo[[idx,max]] if @memo[[idx,max]]
    orig_max = max

    max_score = -Float::INFINITY
    (1..2*max).each do |i|
        next if idx + i > @piles.length
        
        curr_score = @piles[idx...idx+i].sum - score(idx+i,[i,max].max)
        if curr_score > max_score
            max_score = curr_score
            max = i if i > max
        end
    end

    @memo[[idx,orig_max]] = max_score
end


# Nearest Palindrome
def nearest_palindromic(n)
    n = n.to_i
    small = previous_value(n - 1)
    large = next_value(n + 1)

    (n - small) > (large - n) ? large.to_s : small.to_s
end

def previous_value(num)
    str = num.to_s 
    n = str.length
    (0...n).each do |i|
        while str[i] != str[n - 1 - i] # first value != last value
            decrement(str, n - 1 - i)
            return str.to_i if str[0] == '0' # for case 11 where 9 is the answer
        end
    end
    str.to_i
end

def decrement(str, i)
    while str[i] == '0'
        str[i] = '9'
        i -= 1
    end
    str[i] = (str[i].to_i - 1).to_s
end

def next_value(num)
    str = num.to_s
    n = str.length
    (0...n).each do |i|
        while str[i] != str[n - 1 - i]
            increment(str, n - 1 - i)
        end
    end
    str.to_i
end

def increment(str, i)
    while str[i] == '9'
        str[i] = '0'
        i -= 1
    end
    str[i] = (str[i].to_i + 1).to_s
end

# Most Stones Removed with the Same Row or Column
class UnionFind
    def initialize(n)
      @parent = Array.new(n) { |i| i }
      @rank = Array.new(n, 1)
    end
  
    def find(x)
      @parent[x] = find(@parent[x]) if @parent[x] != x
      @parent[x]
    end
  
    def union(x, y)
      root_x = find(x)
      root_y = find(y)
  
      if root_x != root_y
        if @rank[root_x] > @rank[root_y]
          @parent[root_y] = root_x
        elsif @rank[root_x] < @rank[root_y]
          @parent[root_x] = root_y
        else
          @parent[root_y] = root_x
          @rank[root_x] += 1
        end
      end
    end
  end
  
  def remove_stones(stones)
    n = stones.length
    uf = UnionFind.new(n)
  
    row_map = {}
    col_map = {}
  
    stones.each_with_index do |(row, col), i|
      if row_map.key?(row)
        uf.union(i, row_map[row])
      else
        row_map[row] = i
      end
  
      if col_map.key?(col)
        uf.union(i, col_map[col])
      else
        col_map[col] = i
      end
    end
  
    unique_components = (0...n).map { |i| uf.find(i) }.uniq.size
  
    n - unique_components
  end

# Sum of digits of string after convert
def get_lucky(s, k)
    # Map alphabet to int's.
    val = 1
    hash = {}
    ("a".."z").each do |char|
        hash[char] = val
        val += 1
    end

    # Convert s to int but int's are
    # in string form.
    sum = ""
    i = 0
    while i < s.size
        sum += hash[s[i]].to_s
        i += 1
    end
    
    # Transform k times.
    sum = sum.to_s 
    while k > 0
        i = 0
        
        new_sum = 0
        while i < sum.size
            new_sum += sum[i].to_i
            i += 1
        end

        sum = new_sum.to_s
        k -= 1
    end
    
    sum.to_i
end

# Find Missing Observations
def missing_rolls(rolls, mean, n)
    m = rolls.length
    total_sum = mean * (m + n)
    missing_sum = total_sum - rolls.sum
    return [] if missing_sum < n || missing_sum > 6 * n
  
    result = Array.new(n, missing_sum / n)
    (missing_sum % n).times { result[_1] += 1 }
    result
  end

# Delete Nodes from Linked List present in Array
class ListNode
    attr_accessor :val, :next
    def initialize(val = 0, _next = nil)
        @val = val
        @next = _next
    end
end

def modified_list(nums, head)
    num_set = nums.to_set
    dummy = ListNode.new(-1)
    node = dummy

    while head
        unless num_set.include?(head.val)
            node.next = head
            node = node.next
        end
        head = head.next
    end
    node.next = nil
    dummy.next
end

# Minimum bip flips to convert number
def min_bit_flips(start, goal)
    xor = start ^ goal
    xor.to_s(2).count('1')
  end


# XOR Queries of a subarray
def xor_queries(arr, queries)
    n = arr.length
    prefix_xor = Array.new(n + 1, 0)
    
    # Compute the prefix XOR array
    (0...n).each do |i|
      prefix_xor[i + 1] = prefix_xor[i] ^ arr[i]
    end
    
    result = []
    
    # Process each query
    queries.each do |left, right|
      result << (prefix_xor[right + 1] ^ prefix_xor[left])
    end
    
    return result
  end

  # Longest Sub Array with amximum Bitsize AND
  def longest_subarray(nums)
    nums.chunk(&:itself).map(&:last).max.size
  end

  # Different Ways to Add parenthesis
  def initialize
    @memo = {}
  end

  def diff_ways_to_compute(expression)
    return @memo[expression] if @memo.key?(expression)

    result = []
    
    expression.chars.each_with_index do |c, i|
      if ['+', '-', '*'].include?(c)
        left_results = diff_ways_to_compute(expression[0...i])
        right_results = diff_ways_to_compute(expression[(i + 1)..-1])
        
        left_results.each do |left|
          right_results.each do |right|
            case c
            when '+'
              result << left + right
            when '-'
              result << left - right
            when '*'
              result << left * right
            end
          end
        end
      end
    end

    result = [expression.to_i] if result.empty?

    @memo[expression] = result
    result
  end

  # Shortest Palindrome
  def shortest_palindrome(s)
    count = kmp(s.reverse, s)
    s[count..-1].reverse + s
  end
  
  def kmp(txt, patt)
    new_string = patt + '#' + txt
    pi = Array.new(new_string.length, 0)
    i = 1
    k = 0
    while i < new_string.length
      if new_string[i] == new_string[k]
        k += 1
        pi[i] = k
        i += 1
      else
        if k > 0
          k = pi[k - 1]
        else
          i += 1
        end
      end
    end
    pi[-1]
  end

  # Lexicographical Numbers
  def lexical_order(n)
    result = []
    
    (1..9).each do |i|
      dfs(i, n, result)
    end
    
    result
  end
  
  def dfs(current, n, result)
    return if current > n
    
    result << current
    
    (0..9).each do |i|
      next_num = current * 10 + i
      return if next_num > n
      dfs(next_num, n, result)
    end
  end

  # My calendar 2
  class MyCalendarTwo
    def initialize
      @single_booked = []
      @double_booked = []
    end
  
    def intersection(intervals, s, e)
      l = intervals.bsearch_index { |x| x >= s } || intervals.size
      r = intervals.bsearch_index { |x| x > e } || intervals.size
  
      intersection = []
  
      if l.odd?
        if intervals[l] != s
          intersection << s
        else
          l += 1
        end
      end
  
      intersection.concat(intervals[l...r])
  
      if r.odd?
        if intervals[r - 1] != e
          intersection << e
        else
          intersection.pop
        end
      end
  
      intersection
    end
  
    def add(intervals, s, e)
      l = intervals.bsearch_index { |x| x >= s } || intervals.size
      r = intervals.bsearch_index { |x| x > e } || intervals.size
  
      new_intervals = []
      new_intervals << s if l.even?
      new_intervals << e if r.even?
  
      intervals[l...r] = new_intervals
    end
  
    def book(start, end_)
      return false unless intersection(@double_booked, start, end_).empty?
  
      intersection = intersection(@single_booked, start, end_)
  
      unless intersection.empty?
        (0...intersection.size / 2).each do |i|
          i1 = intersection[2 * i]
          i2 = intersection[2 * i + 1]
          add(@double_booked, i1, i2)
        end
      end
  
      add(@single_booked, start, end_)
      true
    end
  end


  # All one data structure
  class AllOne
    def initialize()
        @hash = Hash.new(0)
        @min = ''
        @max = ''
    end


=begin
    :type key: String
    :rtype: Void
=end
    def inc(key)
        @max = key if @hash[key] == @hash[@max]
        @hash[key] += 1
        if @hash[key] == 1
            @min = key
            return
        end
        if @min == key || @min == ''
            min = ''
            val = Float::INFINITY
            @hash.each { |k,v| min,val = k,v if v < val }
            @min = min
        end
    end


=begin
    :type key: String
    :rtype: Void
=end
    def dec(key)
       @min = key if @hash[key] == @hash[@min] && @hash[key] > 1
       @hash[key] -= 1
       @hash.delete(key) if @hash[key] == 0
       if @min == key
            min = ''
            val = Float::INFINITY
            @hash.each { |k,v| min,val = k,v if v < val }
            @min = min
       end
       if @max == key || @max == ''
            max = ''
            val = 0
            @hash.each { |k,v| max,val = k,v if v > val }
            @max = max
        end
    end


=begin
    :rtype: String
=end
    def get_max_key()
        @max
    end


=begin
    :rtype: String
=end
    def get_min_key()
        @min
    end


end
# Make Sum Divisible by P
def min_subarray(nums, p)
    return 0 if nums.sum % p == 0
  
    target = nums.sum % p
    prefix_sum = 0
    prefix_sums_hash = { 0 => -1 }
    min_length = nums.size
    nums.each_with_index do |num, i|
      prefix_sum = (prefix_sum + num) % p
      need_sum = (prefix_sum - target) % p
      min_length = [min_length, i - prefix_sums_hash[need_sum]].min if prefix_sums_hash[need_sum]
      prefix_sums_hash[prefix_sum] = i
    end
    min_length == nums.size ? -1 : min_length
  end


  # Divide Players in Teams of Equal Skill
    def divide_players(skill)
        sorted_array = skill.sort

        pointer1 = 0

        pointer2 = sorted_array.length - 1

        comparer = sorted_array[pointer1] + sorted_array[pointer2]

        totalChemistry = 0

        while ( pointer1 < pointer2)

            number1 = sorted_array[pointer1]

            number2 = sorted_array[pointer2]

            if(number1 + number2 == comparer) 
                totalChemistry += (number1 * number2)
                pointer1 += 1
                pointer2 -= 1
            else 
                return -1
            end
        end
        
        return totalChemistry
    end

    # Minimum add to make parenthesis valid 
    def min_add_to_make_valid(s)
        # Initialize the counter for minimum additions needed
        ans = 0
        
        # Initialize the balance of parentheses (open - close)
        bal = 0
        
        # Iterate through each character in the string
        s.each_char do |ch|
          if ch == '('
            # If it's an opening parenthesis, increment the balance
            bal += 1
          else
            # If it's a closing parenthesis, decrement the balance
            bal -= 1
          end
          
          # If balance becomes negative (more closing than opening parentheses)
          if bal < 0
            # Add the absolute value of balance to answer
            # This represents the number of opening parentheses we need to add
            ans += -bal
            # Reset balance to 0 since we've accounted for the imbalance
            bal = 0
          end
        end
        
        # After processing all characters, add any remaining open parentheses
        # This represents the number of closing parentheses we need to add
        ans += bal
        
        # Return the minimum number of additions needed to make the string valid
        ans
      end


# Number of the smallest unoccupied chair
class Array
    include Comparable

    def add_sorted(v) = insert(bsearch_index {|w| w >= v} || -1, v)
end

class QnCnt
    attr_reader :q, :cnt

    def initialize
        @q, @cnt = [], -1
    end

    def <<(v) = @q.add_sorted(v)

    def shift = (q.empty? ? (@cnt += 1) : @q.shift)

    def first = (q.empty? ? cnt + 1 : q.first)
end

def smallest_chair(times, target_friend)
    eq = QnCnt.new
    times.collect.with_index {|(a, d), id| [a, d, id] }.sort_by(&:first).each_with_object([]) {|(a, d, id), oq|
        eq << oq.shift.last until oq.empty? || oq.first.first > a
        return eq.first if id == target_friend
        oq.add_sorted([d, eq.shift])
    }
end

# Divide Intervals into minimum number of groups
def min_groups a
    h, s = Hash.new(0), 0
    a.each do
        h[_1.first] += 1; h[_1.last + 1] -= 1
    end
    h.sort_by(&:first).map! { s += _1.last } .max
end


# Seperate Black and White Balls
def minimum_steps(s)
    ops = 0
    target = 0
    
    s.each_char.with_index do |char, pos|
      if char == '0'
        ops += pos - target
        target += 1
      end
    end
    ops
  end

# Find the K-th character in string game 1
def kth_character(k)
    result_string = 'a'
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    replacements = 'bcdefghijklmnopqrstuvwxyza'
  
    while result_string.length < k
      current_string = result_string
      new_string = current_string.tr(alphabet, replacements)
      result_string += new_string
    end
  
    result_string[k-1]
  end


  # Flip equivalent tree
  def flip_equiv(root1, root2)
    dfs = lambda do |node1, node2|
      return true if node1.nil? && node2.nil?
      return false if node1.nil? || node2.nil? || node1.val != node2.val
  
      (dfs.call(node1.left, node2.left) && dfs.call(node1.right, node2.right)) ||
        (dfs.call(node1.left, node2.right) && dfs.call(node1.right, node2.left))
    end
    dfs.call(root1, root2)
  end

  # Remove Subfolders
  def remove_subfolders(folder)
    # Sort the folders lexicographically so parent folders come before their subfolders
    folder.sort!
    
    # Initialize result array with the first folder
    ans = [folder[0]]
    
    # Iterate through remaining folders starting from index 1
    (1...folder.length).each do |i|
        # Get the last added folder path and add a trailing slash
        last_folder = ans[-1] + "/"
        
        # Check if current folder starts with last_folder
        # If it doesn't start with last_folder, then it's not a subfolder
        if !folder[i].start_with?(last_folder)
            ans << folder[i]
        end
    end
    
    ans

    # Count square matrices
    def count_squares(matrix)
        # Get dimensions of the matrix
        n = matrix.size        # number of rows
        m = matrix[0].size     # number of columns
        
        # Create a DP table with same dimensions as matrix
        dp = Array.new(n) { Array.new(m, 0) }
        
        # Variable to store total count of squares
        ans = 0
        
        # Initialize first column of DP table
        n.times do |i|
            dp[i][0] = matrix[i][0]
            ans += dp[i][0]
        end
        
        # Initialize first row of DP table
        (1...m).each do |j|
            dp[0][j] = matrix[0][j]
            ans += dp[0][j]
        end
        
        # Fill the DP table for remaining cells
        (1...n).each do |i|
            (1...m).each do |j|
                if matrix[i][j] == 1
                    dp[i][j] = 1 + [dp[i][j-1], dp[i-1][j], dp[i-1][j-1]].min
                end
                ans += dp[i][j]
            end
        end
        
        ans
    end

    # Minimum total distance
    def minimum_total_distance(robot, factory)
        robot.sort!
        factory.sort!
        
        m, n = robot.length, factory.length
        dp = Array.new(m + 1) { Array.new(n + 1, 0) }
        m.times { |i| dp[i][n] = Float::INFINITY }
        
        (n-1).downto(0) do |j|
            prefix = 0
            qq = [[m, 0]]
            
            (m-1).downto(0) do |i|
                prefix += (robot[i] - factory[j][0]).abs
                
                qq.shift if qq[0][0] > i + factory[j][1]
                
                while !qq.empty? && qq[-1][1] >= dp[i][j+1] - prefix
                    qq.pop
                end
                
                qq.push([i, dp[i][j+1] - prefix])
                dp[i][j] = qq[0][1] + prefix
            end
        end
        
        dp[0][0]
    end
    