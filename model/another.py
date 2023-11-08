# a)
# Precondition: 0 <= r0 <= r1 <= len(A) and 0 <= c0 <= c1 <= len(A[0])

# b)
def Helper(r0, r1, c0, c1):
    # Base cases
    if r0 >= r1 or c0 >= c1:
        return False

    # Calculate the "middle" indices.
    i, j = (r0 + r1) // 2, (c0 + c1) // 2

    if A[i][j] == v:
        return True
    elif A[i][j] < v:
        # If the middle element is smaller than v, then we search the
        # bottom-left (rows after i, columns before j) and top-right (rows
        # before i, columns after j) quadrants.
        return Helper(i+1, r1, c0, j) or Helper(r0, i, j+1, c1)
    else:
        # If the middle element is larger than v, then we search the top-left
        # (rows before i, columns before j) and bottom-right (rows after i,
        # columns after j) quadrants.
        return Helper(r0, i, c0, j) or Helper(i+1, r1, j+1, c1)
