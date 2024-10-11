from layer_naive import *


apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

#layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_and_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

#forward
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_and_orange_layer.forward(apple_price, orange_price)
price_with_tax = mul_tax_layer.forward(all_price, tax)

#backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_and_orange_layer.backward(dall_price)
dorange_price, dorange = mul_orange_layer.backward(dorange_price)
dapple_price, dapple = mul_apple_layer.backward(dapple_price)

print(f"Apple price:{apple_price}, Orange price:{orange_price}, All price:{all_price}, Price with tax:{price_with_tax}")
print(f"dApple:{dapple}, dOrange:{dorange}, dApplePrice:{dapple_price}, dOrangePrice:{dorange_price}, dAllPrice:{dall_price}, dTax:{dtax}")