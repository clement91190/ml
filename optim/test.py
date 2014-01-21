class A:
    def __init__(self, truc):
        self.truc = truc


class B(A):
    def __init__(self, truc):
        A.__init__(self, truc)


