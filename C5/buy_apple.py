from layer_naive import *

apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print("Apple price:", apple_price)
print("Total price:", price) # 此处为何不等于 220? 浮点数计算误差导致的误差

# 反向传播
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(f"Dapple price:{dapple_price},dtax:{dtax},dapple:{dapple},dapple_num:{dapple_num}")