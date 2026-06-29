# user = input("Enter the Sentence :").split(" ")
# for i ,name in enumerate(user,start=1):
#     print(f"Word{i} :",name)


n = int(input("enter the number of lines "))

count = {}
for i in range(n):
    user = (input("Enter the line :").lower()).split([" ",",","."])
    for i in user:
        if i not in count:
            count[i] = 1 
        else:
            count[i] +=1 

print(count)