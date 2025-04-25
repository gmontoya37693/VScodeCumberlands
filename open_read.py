# Open the file in read mode, read its content, and print it
f = open('some_file.txt', 'r')
file_data = f.read()
print(file_data)
f.close()

# Open the file in append mode, write a new line, and close it
g = open('some_file.txt', 'a')
g.write('\nHello, this is a new line.\n')  # Write a new line
g.close()

# Reopen the file in read mode to verify the new content
h = open('some_file.txt', 'r')
new_file_data = h.read()
print(new_file_data)
h.close()