import csv

data_list = []
names_list = []

with open("doppelgangers.txt","r") as f:

    for line in f:
        # data = f.readline()
        data = line
        data = data.strip()
        data_list.append([data])

    for index,names in enumerate(data_list):
        # pairing = "Pair_" + str(index+1)
        # append_group = [pairing]
        append_group = []
        names_group = names[0].split(",")
        append_group.append(names_group[0])
        append_group.append(names_group[1].strip())
        names_list.append(append_group)

with open('doppelgangers.csv','w') as f:
    write = csv.writer(f)
    # write.writerow(fields)
    write.writerows(names_list)
print("Done")

