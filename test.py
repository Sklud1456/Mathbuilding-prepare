list=[1,1,1,5,7,41,1,3,4,1,2,4,1,2,5,1,2,4,1]
key=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s']
index=[]
for i in range(len(list)):
    if list[i] <= 4:
        index.append((i,list[i]))
s_key = []
for i in index:
    print(key[i[0]],i[1])
    s_key.append([key[i[0]],i[1]])
print(s_key)