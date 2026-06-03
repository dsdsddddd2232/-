import time
import numpy as np
from numba import njit, prange
N = 4096
RAZMER_BLOKA = 256
# --- вариант 1: обычные три цикла (формула из лекции) ---
@njit(parallel=True)
def variant1_umnozhenie(A, B, C, n):
    for i in prange(n):
        for j in range(n):
            summa = 0.0
            for k in range(n):
                summa = summa + A[i, k] * B[k, j]
            C[i, j] = summa
# --- вариант 2: готовая функция (BLAS) ---
def variant2_umnozhenie(A, B):
    return np.dot(A, B)
# --- вариант 3: по блокам ---
def variant3_umnozhenie(A, B):
    n = N
    C = np.zeros((n, n), dtype=np.float32)
    b = RAZMER_BLOKA
    for i0 in range(0, n, b):
        i1 = i0 + b
        if i1 > n:
            i1 = n
        for k0 in range(0, n, b):
            k1 = k0 + b
            if k1 > n:
                k1 = n
            chast_A = A[i0:i1, k0:k1]
            for j0 in range(0, n, b):
                j1 = j0 + b
                if j1 > n:
                    j1 = n
                C[i0:i1, j0:j1] = C[i0:i1, j0:j1] + np.dot(chast_A, B[k0:k1, j0:j1])
    return C
def main():
    n = N
    print("Автор: Кабров Иван Рагидович")
    print("Группа: 090304-РПИа-о25")
    print("Размер матрицы:", n, "x", n)
    print()
    np.random.seed(42)
    A = np.random.rand(n, n).astype(np.float32)
    np.random.seed(43)
    B = np.random.rand(n, n).astype(np.float32)
    chislo_operaciy = 2 * n * n * n
    print("c = 2*n^3 =", chislo_operaciy)
    print()
    print("Считаем эталон...")
    C_et = variant2_umnozhenie(A, B)
    print()
    print("Вариант 1 (три цикла)...")
    C1 = np.zeros((n, n), dtype=np.float32)
    t_start = time.time()
    variant1_umnozhenie(A, B, C1, n)
    t1 = time.time() - t_start
    p1 = chislo_operaciy / t1 / 1000000
    razn1 = np.max(np.abs(C1 - C_et))
    print("  t =", round(t1, 4), "сек")
    print("  MFlops =", round(p1, 2))
    print("  ошибка =", razn1)
    print()
    print("Вариант 2 (np.dot / BLAS)...")
    t_start = time.time()
    C2 = variant2_umnozhenie(A, B)
    t2 = time.time() - t_start
    p2 = chislo_operaciy / t2 / 1000000
    print("  t =", round(t2, 4), "сек")
    print("  MFlops =", round(p2, 2))
    print()
    print("Вариант 3 (блоки)...")
    t_start = time.time()
    C3 = variant3_umnozhenie(A, B)
    t3 = time.time() - t_start
    p3 = chislo_operaciy / t3 / 1000000
    razn3 = np.max(np.abs(C3 - C_et))
    print("  t =", round(t3, 4), "сек")
    print("  MFlops =", round(p3, 2))
    print("  ошибка =", razn3)
    print()
    print("---------- результат ----------")
    print("вариант 1:  t =", round(t1, 3), "  p =", round(p1, 2), " MFlops")
    print("вариант 2:  t =", round(t2, 3), "  p =", round(p2, 2), " MFlops")
    print("вариант 3:  t =", round(t3, 3), "  p =", round(p3, 2), " MFlops")
    if p2 > 0:
        procent = p3 / p2 * 100
        print("вариант 3 это", round(procent, 1), "% от варианта 2")
if __name__ == "__main__":
    main()
