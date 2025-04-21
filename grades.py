names = input("Enter names separated with commas: ").title.split("'")
assignments = int(input("Enter assignments separated with commas: ").split(","))
grades = int(input("Enter grades separated with commas: ").split(","))

message = "Hi {},\n\nThis is a reminder that you have {} assignments left to \
submit before you can graduate. You're current grade is {} and can increase \
to {} if you submit all assignments before the due date.\n\n"

for name, assignment, grade in zip(names, assignments, grades):
    potential_grade = grade + assignments * 2
    print(message.format(name, assignment, grade, potential_grade))
# This code takes a list of names, assignments, and grades from the user and
# generates a personalized message for each student. The message includes
# the number of assignments left to submit, the current grade, and the potential
# grade if all assignments are submitted before the due date. The code uses
# string formatting to insert the values into the message template.