"""
Assignment 3 Problem 2: Suggest closest feasible chicken nugget order, 
                        and minimum cost combination (2nd Diophantine 
                        equation).
July 24, 2025
German Montoya
"""

# Define box sizes and their costs
def find_combinations(n):
    combos = []
    for a in range(n // 6 + 1):
        for b in range(n // 9 + 1):
            for c in range(n // 22 + 1):
                if 6 * a + 9 * b + 22 * c == n:
                    combos.append((a, b, c))
    return combos

# Function to find the minimum cost combination
def find_min_cost(combos):
    min_cost = float('inf')     # Start with a large integer
    min_combo = None
    for a, b, c in combos:
        cost = 3 * a + 4 * b + 9 * c
        if cost < min_cost:
            min_cost = cost
            min_combo = (a, b, c)
    return min_cost, min_combo

# Function to find the closest feasible order
def find_closest_feasible(order):
    offset = 1
    while True:
        for direction in [-1, 1]:
            alt_order = order + direction * offset
            if alt_order > 0:
                combos = find_combinations(alt_order)
                if combos:
                    return alt_order, combos
        offset += 1

# start of the main program and user interaction
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
        if combos:  # If there are valid combinations, find the minimum cost and display it
            min_cost, min_combo = find_min_cost(combos)
            print(f"The minimum cost for {order} chicken nuggets is ${min_cost}:")
            print({'Six piece': min_combo[0], 'Nine piece': min_combo[1], 'Twenty two piece': min_combo[2]})
        else:
            closest_order, closest_combos = find_closest_feasible(order)
            min_cost, min_combo = find_min_cost(closest_combos)
            print(f"We don't have an exact match for {order} nuggets.")
            print(f"The closest feasible order is {closest_order} nuggets.")
            print(f"The minimum cost for {closest_order} chicken nuggets is ${min_cost}:")
            print({'Six piece': min_combo[0], 'Nine piece': min_combo[1], 'Twenty two piece': min_combo[2]})

