print("This is a simple example of using with to open a file.")
def create_cast_list(filename):
    cast_list = []
    #use with to open the file filename
    with open(filename) as a:
    #use the for loop syntax to process each line
        for line in a:
            name = line.split(',')[0]
            #and add the actor name to cast_list
            cast_list.append(name)

    return cast_list

cast_list = create_cast_list('flying_circus_cast.txt')
for actor in cast_list:
    print(actor)