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
