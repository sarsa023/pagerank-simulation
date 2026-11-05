import numpy as np
import functions as f
import matplotlib.pyplot as plt


# 1
print("Problem 1:")
print("Matrix Construction ( Finding A_hat ) \n")
print("Original matrix: ")
# Test with figure 13.5
A = np.array([
    [0, 0, 0, 0],
    [1, 0, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0]
])
labels = ['a', 'b', 'c', 'd']
print(np.array(labels))
print(A)
# Must give back:
# [
#     [0, 1 / 4, 0, 0],
#     [1 / 3, 1/ 4, 1 / 2, 0],
#     [1 / 3, 1 / 4, 0, 1],
#     [1 / 3, 1 / 4, 1 / 2, 0]
# ]
print("Modified array to A_hat:")
new_A = f.process_matrix(A, labels=labels)
print(new_A)

print("\n******************************\n")

# 2
print("Problem 2:")
print("computing Page Ranks methods:")
print("((damping factor defaulting to 0.85))")
print("\n1. Solving a Linear System method:\n")
#Save values
p_lin = f.linsolve(new_A, labels=labels)
print(p_lin)
print("\n2. Finding Eigen Values method:\n")
p_eig = f.eigensolve(new_A, labels)
print(p_eig)
print("\n3. The Iterative method:\n")
p_iter = f.itersolve(new_A, labels=labels)
print(p_iter)

print("\n** Convergence check **")
print("\nlinsolve vs eigensolve:")
f.compare_methods(p_lin, p_eig)

print("\n\nlinsolve vs itersolve:")
f.compare_methods(p_lin, p_iter
                  )
print("\n\neigensolve vs itersolve:")
f.compare_methods(p_eig, p_iter)

print("\n\nAll three PageRank methods (linear solve, eigenvector solve, and iterative method)\nproduced matching results within a small numerical tolerance (not exceeding 1e-6).\nThis confirms both convergence and correctness.\n")

print("\n*****************************\n")

# 3
print("Problem 3:")
print("Sorting Pages from highest to lowest page rank:")
print(f.arrangement(p_lin))

plt.figure(figsize=(15, 7))

#data represented in pie chart
plt.subplot(1, 2, 1)
plt.pie(p_lin.values(), labels=labels, autopct='%1.1f%%', startangle=90)
plt.title("Figure 13.5 data pie chart")

print("\n*****************************\n")

# 4
print("Problem 4:")
print("Web Pages file processing")
# Parsing file
file_name = r'web_pages.txt'
file = open(file_name)
lines = file.readlines()
file.close()

# Since each node has its own line:
n = len(lines)

# matrix start from all zero
adj_matrix = [[0 for _ in range(n)] for _ in range(n)]

# process line by line
for line in lines:

    # skipping empty lines
    if line.strip() == "":
        continue

    nums = list(map(int, line.strip().split('/')))  # .strip() to remove whitespace then split based on '/'
    node = nums[0]                                  # then map() applies int() casting to all elements
    for x in nums[1:]:                              # "2" -> 2 and so on
        adj_matrix[x-1][node-1] = 1                 # then assign 1 on this node's **column** for each node it points to

adj_matrix = np.array(adj_matrix)
print("\nAdjacency matrix from file :\n")
print(adj_matrix)

print("\nProcessed Matrix:\n")
# set labels
file_labels = [i + 1 for i in range(n)]
new_adj = f.process_matrix(adj_matrix, file_labels)
print(new_adj)
print("One can see that max amount of pages linked to from 1 page is 4\nSo minimum nonzero value in our processed matrix is 0.25\n")

# Finding page ranks (damping factor defaulted to 0.85)
file_ranks = f.linsolve(new_adj, file_labels)
print("Pages ranks:")
rounded = {key : round(val, 6) for key, val in file_ranks.items()} # for better formatting
print(rounded)

print("\nPages sorted by their ranks:")
print(f.arrangement(file_ranks))

plt.subplot(1, 2, 2)
vals = [v * 1000 for v in file_ranks.values()] # scaling for better observing
plt.bar(file_labels, vals)
plt.title("File Web Pages Ranks")
plt.xlabel("Pages")
plt.ylabel("Rank x 1000")
plt.show()