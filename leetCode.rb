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