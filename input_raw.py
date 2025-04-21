names = []
assignments = []
grades = []

num_students = 3

while student <= num_students:
    name = input("Enter student name {}: ".format(student))
    names.append(name)
    assignment = input("Enter assignment name {}: ".format(student))
    assignments.append(assignment)
    grade = input("Enter grade for {}: ".format(name))
    grades.append(grade)
    student += 1

    print(names)
    print(assignments)
    print(grades)