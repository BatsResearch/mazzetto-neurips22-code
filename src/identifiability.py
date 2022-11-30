# M: C x A matric containing (1/0 = (*)/-1)
import numpy as np

# def identifiability(M):
#     k, n = np.shape(M)
#     # C, A = np.shape(M)
#     gamma = [set(range(k)) for c in range(k)]
#     #gamma = [set(range(C)) for c in range(C)]
    
#     for j in range(k):
#     # for c in range(C):
#         for i in range(n):
#         # for a in range(A):
#             # se m[c,a] e' * non c'e' modo che io possa 
#             #distinguerlo tra classi
#             if M[j,i] == 0:
#             #if M[c,a] == 0:
#                 continue
#             # Altrimenti, vedo quali classi sono tra loro
#             # indistinguibili rispetto all'attributo
#             tmp = set()
#             for l in range(k):
#             #for cls in range(C):
#                 if M[l,i] == M[j,i] or M[l,i] == 0:
#                 #if M[cls,a] == M[c,a]:# or M[l,i] == 0:
#                     tmp.add(l)
#             # tmp e' il set di classi per cui la classe c
#             # non puo' essere distinta
#             gamma[j] = gamma[j] & tmp
            
#     return gamma

def identifiability(M):
    #k, n = np.shape(M)
    C, A = np.shape(M)
    #gamma = [set(range(k)) for c in range(k)]
    gamma_quick = [set(range(C)) for c in range(C)]

    #for j in range(k):
    for c in range(C):
        #for i in range(n):
        for a in range(A):

            col_cls = M[:,a]
            val_c = col_cls[c]

            if val_c == 0:
                continue

            tmp = set(list(np.where((col_cls == val_c) | (col_cls == 0))[0]))

            gamma_quick[c] = gamma_quick[c] & tmp
    
    return gamma_quick


def identifiability_continuous(M, t):
    C, A = np.shape(M)

    gamma_quick = []
    for c in range(C):
        neigh = []
        for cls in range(C):
            diff_below_th = np.where(np.abs(M[c] - M[cls]) <= t)[0]
            if len(diff_below_th) == (A):
                neigh += [cls]

        gamma_quick += [neigh]

    return gamma_quick