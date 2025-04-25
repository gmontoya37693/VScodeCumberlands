f = open('some_file.txt')
file_data = f.read()
print(file_data)
f.close()
# The above code opens a file named 'some_file.txt', reads its content, and prints it to the console.
# The file is then closed.

f = open('some_file.txt', 'w')
f.write('Hello, Germanchirris!' + '!')
print(file_data)
f.close()