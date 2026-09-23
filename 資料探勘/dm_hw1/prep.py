import csv

input_path = 'hw1.csv'
output_path = 'hw1_afterp.csv'

with open(input_path, 'r') as csv1:
    reader = csv.reader(csv1)
    with open(output_path, 'w', newline='') as csv2:
        writer = csv.writer(csv2)
        for row in reader:
            new_row = [item.replace("'", "") for item in row]
            final_row = [item.replace("False", "?") for item in new_row]
            writer.writerow(final_row)