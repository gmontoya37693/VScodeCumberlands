f = open('some_file.txt')
file_data = f.read()
print(file_data)
f.close()
# The above code opens a file named 'some_file.txt', reads its content, and prints it to the console.
# The file is then closed.

g = open('some_file.txt', 'a')
file_data = g.write('\nHello, this is a new line.\n')
print(file_data)
g.close()