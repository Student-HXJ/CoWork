class TreeNode:
    def __init__(self, keyname=None, value=None):
        self.key = keyname
        self.value = value
        self.lchild = None
        self.rchild = None


class HuffmanTree(object):
    def __init__(self, binstr):
        hflist = self.get_hflist(binstr)
        self.a = [TreeNode(item[0], item[1]) for item in hflist]
        while len(self.a) != 1:
            self.a.sort(key=lambda x: x.value, reversed=True)
            root = TreeNode(self.a[-1].value + self.a[-2].value)
            root.lchild = self.a.pop(-1)
            root.rchild = self.a.pop(-1)
            self.a.append(root)
        self.root = self.a[0]

    def get_hflist(self, binstr):
        hfdic = {}
        for ch in binstr:
            if ch in hfdic.keys():
                hfdic[ch] += 1
            else:
                hfdic[ch] = 1
        hflist = list(hfdic.items())
        return hflist

    def get_len(self):

    def pre(self, tree, length):
        if not tree:
            return

        self.b[length] = 0
        self.pre(tree.lchild, length + 1)
        self.pre(tree.rchild, length + 1)

    def get_code(self):
        self.pre(self.root, 0)


if __name__ == "__main__":
    binstr = "0900e9aaa8e7b2bee781b5a25d4f00000200a35d4f00fbf9930e0080064300000000ae47bb411500e9aaa8e7b2bee781b5e79a84e5b08fe58fafe788b101000200"
