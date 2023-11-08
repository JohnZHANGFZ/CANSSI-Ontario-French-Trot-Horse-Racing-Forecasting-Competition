a = [[0.5,0.4,0.1],[0.5,0.3,0.2]]
b = [[0.5,0.4,0.1],[0.6,0.2,0.1]]

# check total probability = 1
def test_prob_sum(data):
    total_prob = 0
    for i in range(0, len(data)):
        for j in range(0, len(data[i])):
            total_prob += data[i][j]
        if abs(total_prob - 1) < 0.001:
            total_prob = 0
        else:
            return False
    return True


# check if length of real data = that of test data
def test_length(real_data, test_data):
    if len(real_data) != len(test_data):
        return False
    for i in range(0, len(real_data)):
        if len(real_data[i]) != len(test_data[i]):
            return False
    return True


# check deviation/accuracy
def test_deviation(real_data, test_data):
    total_deviation = 0
    for i in range(0, len(test_data)):
        for j in range(0, len(test_data[i])):
            a = (real_data[i][j] - test_data[i][j])/len(real_data[i])
            total_deviation += a
    return total_deviation / len(real_data)



# change rank to probability
def insert(L, i, indexlist):
    value = L[i]
    value_index = indexlist[i]
    while i>0 and L[i-1] > value:
        L[i] = L[i-1]
        indexlist[i] = indexlist[i-1]
        i = i - 1
    L[i] = value
    indexlist[i] = value_index


def insertion_sort(L, indexlist):
    for i in range(len(L)):
        insert(L, i, indexlist)
    #print(L)
    #print(indexlist)


def prob(index_list):
    cur_prob = 0.6
    prob_list = []
    for i in range(len(index_list)):
        if i < len(index_list) - 1:
            prob_list.append(cur_prob)
            cur_prob = (1-sum(prob_list))/2
        else:
            prob_list.append(1-sum(prob_list))
    return prob_list


def to_prob(rank):
    result = []
    for i in range(0, len(rank)):
        predicted = []
        index_list = []
        probability = []
        for j in range(0, len(rank[i])):
            predicted.append(rank[i][j])
            index_list.append(j+1)
        probability = prob(index_list)
        insertion_sort(predicted, probability)
        result.append(predicted)
        result.append(probability)
    return result










