import math
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
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
                self.grad  += out.grad     # 我拿多少，上家就拿多少
                other.grad += out.grad
        out._backward = _backward
        return out
    __radd__ = __add__          # 1 + a  →  a + 1
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad  += other.data * out.grad   # 我拿"对方的数值"份
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out
    __rmul__ = __mul__          # 2 * a  →  a * 2
    def __neg__(self):              # -a
        return self * -1

    def __sub__(self, other):       # a - b
        return self + (-other)

    def __rsub__(self, other):      # 1 - a
        return (-self) + other
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "只支持 int/float 次方"
        out = Value(self.data ** other, (self,), f'**{other}')
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)     # 前向：算出激活值
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad            # 局部导数 = 1 - tanh²
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
# a, b, c = Value(2.0), Value(-3.0), Value(10.0)
# e = a * b
# d = e + c

# d.backward()          # 一句话就够，不再手动调

# print(a.grad, b.grad, c.grad)      # 仍然是 -3.0 2.0 1.0
# a = Value(2.0)
# b = Value(3.0)

# print(a - b)     # ?
# print(a - 1)     # ?
# print(1 - a)     # ?
# print(-a)        # ?

# d = a - b
# d.backward()
# print(a.grad, b.grad)
# a = Value(3.0)
# y = a ** 2
# print(y)
# y.backward()
# print(a.grad)
# x = Value(0.0)
# y = x.tanh()
# y.backward()
# print(y)          # ?
# print(x.grad)     # ?
h = 1e-6
x = 1.0
print("数值导数:", (math.tanh(x+h) - math.tanh(x-h)) / (2*h))

xv = Value(1.0)
yv = xv.tanh()
yv.backward()
print("autograd:", xv.grad)