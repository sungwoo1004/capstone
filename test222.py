input_arr = [150,266,427]
a=1
a_fre = [0]*8
for num in input_arr:
    a *= num
    
a_str = str(a)
print(a_str)
a_len = len(a_str)
for b in range(a_len):
    last_digit = a % 10
    a = a // 10
    print(a, last_digit)

    a_fre[last_digit] = a_fre[last_digit] + 1
    
print(a_fre)

