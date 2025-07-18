def apply_to_each(L, f):
    """ Assumes L is a list, f a function
Mutates L by replacing each element, e, of L with f(e) """
    for i in range(len(L)):
        L[i] = f(L[i])

L = [1, -2, 3.33]
print('L =', L)
print('Apply abs to each element of L.')
apply_to_each(L, abs)
print('L =', L)
print( 'Apply int to each element of', L)
apply_to_each(L, int)
print('L =', L)
print('Apply squaring to each element of', L)
apply_to_each(L, lambda x: x**2)
print('L =', L)