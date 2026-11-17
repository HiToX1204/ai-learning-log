class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None      # 占位：默认不做任何事

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
                self.grad  += out.grad     # 我拿多少，上家就拿多少
                other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad  += other.data * out.grad   # 我拿"对方的数值"份
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out
    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)          # 先标记，防止重复
                for child in v._prev:   # 递归处理所有父节点
                    build(child)
                topo.append(v)          # ← 关键：所有父节点处理完，才把自己加进去

        build(self)
        print("topo 顺序:", [str(v.data) for v in topo])    # ← 临时加
        self.grad = 1.0                 # 种子：输出对自己的梯度是 1

        for node in reversed(topo):     # 逆序：从输出往回
            node._backward()
# a, b, c = Value(2.0), Value(-3.0), Value(10.0)
# e = a * b
# d = e + c

# print("初始：", d.grad, e.grad, c.grad)   # 全 0

# d.grad = 1.0
# print("设完 d.grad：", d.grad, e.grad, c.grad)   # 1.0 0 0

# d._backward()
# print("d 反传后：", d.grad, e.grad, c.grad)      # 1.0 1.0 1.0

# e._backward()
# print("e 反传后：", a.grad, b.grad, e.grad)      # -3.0 2.0 1.0
# def f(a, b, c):
#     return a * b + c

# h = 1e-6
# print((f(2+h, -3, 10) - f(2-h, -3, 10)) / (2*h))   # ∂d/∂a
# print((f(2, -3+h, 10) - f(2, -3-h, 10)) / (2*h))   # ∂d/∂b
# print((f(2, -3, 10+h) - f(2, -3, 10-h)) / (2*h))   # ∂d/∂c
a, b, c = Value(2.0), Value(-3.0), Value(10.0)
e = a * b
d = e + c

d.backward()          # 一句话就够，不再手动调

print(a.grad, b.grad, c.grad)      # 仍然是 -3.0 2.0 1.0