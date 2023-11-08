def Pow(x, y) :
    z = 1
    while y > 0 :
        if y % 2 == 1 :
            z = z * x
        x = x * x
        y = y // 2
    return z


print(Pow(2, 3))


# a)
# Precondition: x ∈ R, y ∈ N.
# Post-condition: return xy (return 1 if x = y = 0).
def Pow1(x, y):
    # Precondition: zi * xi^yi = x^y, xi, yi, zi are Natural numbers
    # Post-condition: return xi^yi (return 1 if xi = yi = 0).
    def r(xi, yi, zi):
        if yi == 0:
            return zi
        if yi % 2 == 1:
            return r(xi ** 2, yi // 2, zi * xi)
        if yi % 2 == 0:
            return r(xi ** 2, yi // 2, zi)

    # Initial arguments: x, y, and 1 (for z)
    return r(x, y, 1)


# b) Precondition: xi, yi, zi are Natural numbers (with xi being in R for
# generality).
# Post-condition: return zi * xi^yi.
def r(xi, yi, zi):
    if yi == 0:
        return zi
    if yi % 2 == 1:
        return r(xi ** 2, yi // 2, zi * xi)
    else:
        return r(xi ** 2, yi // 2, zi)


# Precondition: x ∈ R, y ∈ N.
# Post-condition: return x^y (return 1 if x = y = 0).
def Pow2(x, y):
    return r(x, y, 1)


# Precondition: x ∈ R, y ∈ N.
# Post-condition: return x^y (return 1 if x = y = 0).
def PowR(x, y):
    if y == 0:
        return 1
    elif y % 2 == 1:
        return x * PowR(x * x, y // 2)
    else:
        return PowR(x * x, y // 2)
