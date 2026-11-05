import numpy as np

def process_matrix(A, labels=None): #defaulting labels to None
    A = np.array(A, dtype=float) #ensuring type

    # Check if square
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square")

    n = A.shape[0]

    # Use default labels if none
    if labels is None:
        labels = list(range(n))

    if len(labels) != n:
        raise ValueError("Number of labels must match number of nodes")

    # Fix sinks (columns with all 0s)
    for j in range(n):
        if np.sum(A[:, j]) == 0:
            A[:, j] = 1  # fill column with ones

    # Normalise columns
    A = A / A.sum(axis=0)
    return A


# each function here accepts processed array A^
# (no sink nodes and the array has been normalized:
# columns sum up to 1)
#functions return {label : rank} each

def linsolve(A_hat, labels, eps=0.85): # eps : damping factor
    A_hat = np.array(A_hat, dtype=float)  # ensuring type

        # equation to solve:
        # (I - epsA^)p = (1 - eps) / n * 1
        # we need :
        # I, an identity matrix of size n x n
        # 1 a vector of ones, size n x 1 (column vector)
        # p is the n x 1 column vector of page ranks
        # transpose of p is what we want

    n = len(labels) # labels size is our number of pages
    I = np.eye(n) # get an identity matrix of size n x n
    ones_vector = np.ones((n, 1)) # create column vector of ones

    # now we use linalg.solve, the way it works for Ax = b
    # takes left side A, right side b and returns the solution x
    # linalg.solve(A, b) : x
    # x will be of same shape as b
    # in our case p will be of same shape as our ones vector (a column vector)

    l = I - eps * A_hat
    r = (1 - eps) / n * ones_vector
    p = np.linalg.solve(l, r)
    p = p.flatten() # turn from n x 1 to 1 x n to use in dictionary

    return {labels[i] : float(p[i]) for i in range(n)}

def eigensolve(A_hat, labels, eps=0.85):
    A_hat = np.array(A_hat, dtype=float)

        # idea here is that after some mathematical manipulation we get that for:
        # B = epsA^ + (1 - eps) / n * E where E is a n x n matrix of ones
        # -> Bp = p
        # meaning that p is an eigenvector of B with eigenvalue 1
        # we want to find B then use linalg.eig

    n = len(labels)
    E = np.ones((n, n)) # n x n matrix of ones
    B = eps * A_hat + (1 - eps) / n * E

    # linalg.eig returns eigen values and eigen vectors for all possible sols
    # there can be complex solutions, we want to filter out real parts
    # and according to Perrons's theorem mentioned in project description:
    # since B's entries are all > 0 and its columns sum up to 1
    # there will be one and only real solution with its value 1

    eig_vals, eig_vectors = np.linalg.eig(B)
    eig_vals = np.real(eig_vals) # take only real parts of sols
    # function argmax returns the ((index)) of max value
    # complex sols real part will be less than 1 so we are sure this will give us the real sol index
    i = np.argmax(eig_vals)

    # we want this index because each column vector in eigen vectors returned corresponds to the value at this columns index
    p = np.real(eig_vectors[:, i]) # throwing away imaginary part (even if 0)

    p = p / np.sum(p) # normalizing p to make it sum to 1
    p = p.flatten() # turn to 1 x n

    return {labels[i] : float(p[i]) for i in range(n)}

def itersolve(A_hat, labels, eps=0.85):
    A_hat = np.array(A_hat, dtype=float)

    # this method is about finding p(t + 1) from p(t) from the original equation:

    # p(t + 1) = eps * A^ * p(t) + (1 - eps) / n * 1

    # doing this until ||p(t + 1) - p(t)||1 is negligible

    # ||p(t + 1) - p(t)||1 means a vector norm L1:
    # sum of absolute differences between each corresponding elements in p(t + 1) and p(t)
    # and 1 here is for it being L1 norm as there are other types
    # linalg.norm computes this, but we need to define the norm type as mentioned

    # we also want to use np.matmul function for multiplying the 2 2d arrays
    # this function in numpy can be implemented using the @ sign

    # let's set the value we want our vector norm to be less than to 1e-6
    v = 1e-6

    # and set initial p value to all 1 / n
    # such that at the beginning and before clicking any link all pages have the same probability of being clicked into
    n = len(labels)
    p = np.full((n, 1), 1 / n) # n x 1 vector with values 1 / n
    ones = np.ones((n, 1))

    # preparing for loop
    p_next = eps * np.matmul(A_hat, p) + (1 - eps) / n * ones

    # but also to avoid iterating for too long for larger matrices let's set a bound
    bound = 1000
    t = 0
    while np.linalg.norm(p_next - p, 1) >= v and t < bound: # ord 1 in norm functions denotes L1 norm
        p = p_next
        p_next = eps * np.matmul(A_hat, p) + (1 - eps) / n * ones
        t += 1

    p = p_next.flatten()
    return {labels[i] : float(p[i]) for i in range(n)}

# convergence check
def compare_methods(p1, p2):
    print("Differences:")
    for key in p1:
        print(abs(p1[key] - p2[key]), end=" ")

#function to arrange value of page rank
def arrangement(page_rank):
    return sorted(list(page_rank.keys()), key= lambda x: -page_rank[x])

# lb = ['a', 'b', 'c', 'd']
# Ah = np.array([
#     [0, 1 / 4, 0, 0],
#     [1 / 3, 1/ 4, 1 / 2, 0],
#     [1 / 3, 1 / 4, 0, 1],
#     [1 / 3, 1 / 4, 1 / 2, 0]
# ], dtype=float)
# print(linsolve(Ah, labels=lb))
# print(eigensolve(Ah, labels=lb))
# print(itersolve(Ah, labels=lb))
# p = linsolve(Ah, labels=lb)
#
# print(arrangement(p))


