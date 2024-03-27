

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