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


 