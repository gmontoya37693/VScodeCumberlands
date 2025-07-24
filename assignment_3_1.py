"""
Assignment 3 Problem 1: quantities of chicken nuggets that fit
within available order quantities
July 23, 2025
German Montoya
"""

while True:
    test = input("Do you want to buy chicken nuggets? (yes/no): ").lower()
    if test == "no":
        print("Thank you for coming.")
        break
    elif test == "yes":
        # Prompt exactly as required
        print("How many chicken nuggets would you like to order? ", end="")
        order = int(input())
        n_combinations = {}
        n_combinations[order] = []
        for a in range(order // 6 + 1):
            for b in range(order // 9 + 1):
                for c in range(order // 22 + 1):
                    n = 6 * a + 9 * b + 22 * c
                    if n == order:
                        n_combinations[order].append((a, b, c))

        # Output results
        if n_combinations[order]:
            print(f"For an order size of {order}, choose from the following {len(n_combinations[order])} option(s):")
            for combo in n_combinations[order]:
                print({'Six piece': combo[0], 'Nine piece': combo[1], 'Twenty two piece': combo[2]})
        else:
            print(f"\nSorry, you cannot order exactly {order} chicken nuggets with boxes of 6, 9, and 22.")


