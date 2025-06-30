while True:
    user_input = input("Input a number to find its square root: ")
    try:
        number = int(user_input)
        break
    except ValueError:
        print("Please enter a valid integer.")
# Now you can use 'number' as an integer