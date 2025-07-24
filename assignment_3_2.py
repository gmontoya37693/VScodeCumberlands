"""
Assignment 3 Problem 2: Suggest closest feasible chicken nugget order
July 24, 2025
German Montoya
"""

def find_combinations(n):
    combos = []
    for a in range(n // 6 + 1):
        for b in range(n // 9 + 1):
            for c in range(n // 22 + 1):
                if 6 * a + 9 * b + 22 * c == n:
                    combos.append((a, b, c))
    return combos

while True:
    # Prompt user for input
    test = input("Do you want to buy chicken nuggets? (yes/no): ").lower()
    if test == "no":
        print("Thank you for coming.")
        break
    # Prompt user for order quantity when they answer "yes"
    elif test == "yes":
        order = int(input("How many chicken nuggets would you like to order? "))
        combos = find_combinations(order)
        if combos:  # If there are valid combinations, show them
            print(f"For an order size of {order}, choose from the following {len(combos)} option(s):")
            for combo in combos:
                print({'Six piece': combo[0], 'Nine piece': combo[1], 'Twenty two piece': combo[2]})
        else:
            # Find the closest feasible n (search up and down)
            offset = 1      # set initial offset to 1
            found = False
            while not found:
                for direction in [-1, 1]:   # search both directions, iterating over only those two values
                    alt_order = order + direction * offset
                    if alt_order > 0:   # Ensure positive order (I admit this is COPILOT's suggestion; very clever though)
                        alt_combos = find_combinations(alt_order)
                        if alt_combos:
                            print(f"\nSorry, you cannot order exactly {order} chicken nuggets with boxes of 6, 9, and 22.")
                            print(f"The closest feasible quantity is {alt_order}, with the following option(s):")
                            for combo in alt_combos:
                                print({'Six piece': combo[0], 'Nine piece': combo[1], 'Twenty two piece': combo[2]})
                            found = True
                            break   # Exit the loop if a combination is found
                offset += 1 # Increment offset to search further away if no combination found