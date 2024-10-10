import numpy as np


class GateCircuitSimulation:

    def __init__(self):
        pass

    # 模拟与门
    def AND(self, x1, x2):
        x = np.array([x1, x2])
        w = np.array([0.5, 0.5])
        b = -0.7
        tmp = np.sum(w * x) + b

        if tmp <= 0:
            return 0
        else:
            return 1

    def OR(self, x1, x2):
        x = np.array([x1, x2])
        w = np.array([0.5, 0.5])
        b = - 0.2
        tmp = np.sum(w * x) + b
        if tmp <= 0:
            return 0
        else:
            return 1

    def NAND(self, x1, x2):
        x = np.array([x1, x2])
        w = np.array([-0.5, -0.5])
        b = 0.7
        tmp = np.sum(w * x) + b
        if tmp <= 0:
            return 0
        else:
            return 1

    def XOR(self, x1, x2):
        s1 = self.OR(x1, x2)
        s2 = self.NAND(x1, x2)
        return self.AND(s1, s2)


gate_circuit = GateCircuitSimulation()
print("#### AND ######")
print(gate_circuit.AND(1, 1))
print(gate_circuit.AND(0, 1))
print(gate_circuit.AND(1, 0))
print(gate_circuit.AND(0, 0))


print("#### OR ######")
print(gate_circuit.OR(0, 0))
print(gate_circuit.OR(0, 1))
print(gate_circuit.OR(1, 0))
print(gate_circuit.OR(1, 1))

print("#### NAND ######")
print(gate_circuit.NAND(0, 0))
print(gate_circuit.NAND(0, 1))
print(gate_circuit.NAND(1, 0))
print(gate_circuit.NAND(1, 1))

print("#### XOR ######")
print(gate_circuit.XOR(0, 0))
print(gate_circuit.XOR(0, 1))
print(gate_circuit.XOR(1, 0))
print(gate_circuit.XOR(1, 1))

