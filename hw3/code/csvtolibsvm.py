import csv
import argparse

def libsvm_line(label, line):
    new_line = ""
    new_line = new_line + label

    for i, value in enumerate(line):
        level = "%s:%s" % (i+1, value)
        new_line = new_line + " "
        new_line = new_line + level
    new_line = new_line + '\n'
    return new_line

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csvpath', help="selected csv filepath")
    parser.add_argument('--libsvmpath', help="selected filepath for converted csv to write to")
    args = parser.parse_args()

    csvf = args.csvpath
    libsvmf = args.libsvmpath

    with open(csvf, 'r') as csvfile, open(libsvmf, 'w') as libsvmfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            label = row.pop(0)
            line = libsvm_line(label, row)
            libsvmfile.write(line)

if __name__ == '__main__':
    main()
