names = []
assignments = []
grades = []
student = 1
num_students = 4

while student <= num_students:
    name = input("Enter student name {}: ".format(student))
    names.append(name)
    assignment = int(input("Enter number of assignments left for {}: ".format(name)))
    assignments.append(assignment)
    grade = int(input("Enter grade for {}: ".format(name)))
    grades.append(grade)
    student += 1

for n in range(num_students):
    potential_grade = grades[n] + (assignments[n] * 2)  # Example calculation
    print(
        "Hi {},\n\nThis is a reminder that you have {} assignments left to "
        "submit before you can graduate. Your current grade is {} and it can increase "
        "to {} if you submit all assignments before the due date.\n\n".format(
            names[n], assignments[n], grades[n], potential_grade
        )
    )