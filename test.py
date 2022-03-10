f = open("./model/high.txt", "r")
high = int(f.read())
print("num:", high)
f.close()

f = open("model/high.txt", "w")
f.write(str(24))
f.close()