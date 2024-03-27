

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