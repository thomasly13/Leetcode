# Merge Intervals

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