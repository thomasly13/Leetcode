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