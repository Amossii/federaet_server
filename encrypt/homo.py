import random
from Crypto.Util.number import getPrime
def creatK():
    bit_length=4
    p = getPrime(bit_length)
    q = getPrime(bit_length)
    n=p*q
    r = random.randint(n + 1, 2 * n)
    k=r*n
    return n,k

class SimpleAdditiveHomomorphic:
    def __init__(self):
        bit_length = 4
        p = getPrime(bit_length)
        q = getPrime(bit_length)
        n = p * q
        r = random.randint(n + 1, 2 * n)
        k = r * n
        self.n = n
        self.k = k
        print('k:',self.k)

    def encrypt(self, plaintext):
        # 加密明文 x 为 x + k
        ciphertext = (plaintext + self.k) % (self.n ** 2)  # 使用 n^2 作为模数以支持同态性质
        return ciphertext

    def decrypt(self, ciphertext):
        # 解密密文 y 为 (y - k) % n
        return (ciphertext - self.k) % self.n

    def add(self, ciphertext1, ciphertext2):
        # 对两个密文进行加法
        return (ciphertext1 + ciphertext2) % (self.n ** 2)
    def getK(self):
        return self.k

# 示例
crypto = SimpleAdditiveHomomorphic()

plaintext1 = 0.00004
plaintext2 = 0.002
plaintext3 = 1
# 加密两个数
ciphertext1 = crypto.encrypt(plaintext1)
ciphertext2 = crypto.encrypt(plaintext2)
ciphertext3=crypto.encrypt(plaintext3)
# 同态加法
added_ciphertext = crypto.add(ciphertext1, ciphertext2)
added_ciphertext=crypto.add(added_ciphertext,ciphertext3)

# 解密结果
decrypted_result = crypto.decrypt(added_ciphertext)

print(decrypted_result)