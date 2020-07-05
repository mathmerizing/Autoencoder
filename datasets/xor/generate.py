from random import randint
with open('./input.txt','w') as file1:
    with open('./output.txt','w') as file2:
        for i in range(1000):
            # [a,b] = [randint(0,1), randint(0,1)]
            b = i % 2
            a = int((i % 4) > 1)     
            file1.write(str(a) + " " +str(b) + '\n')
            file2.write(str(a ^ b) +'\n')
