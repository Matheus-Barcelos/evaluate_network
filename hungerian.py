import numpy as np

def min_zero_row(zero_mat, mark_zero):
    
    '''
    The function can be splitted into two steps:
    #1 The function is used to find the row which containing the fewest 0.
    #2 Select the zero number on the row, and then marked the element corresponding row and column as False
    '''

    #Find the row
    min_row = [99999, -1]

    for row_num in range(zero_mat.shape[0]): 
        num_zeros = np.sum(zero_mat[row_num] == True)
        if num_zeros > 0 and min_row[0] > num_zeros:
            min_row = [num_zeros, row_num]

    # Marked the specific row and column as False
    zero_index = np.where(zero_mat[min_row[1]] == True)[0][0]
    mark_zero.append((min_row[1], zero_index))
    zero_mat[min_row[1], :] = False
    zero_mat[:, zero_index] = False
    ...

def mark_matrix(cost_matrix:np.ndarray, cur_mat:np.ndarray):

    '''
    Finding the returning possible solutions for LAP problem.
    '''

    #Transform the matrix to boolean matrix(0 = True, others = False)
    zero_bool_mat = (cur_mat == 0)
    zero_bool_mat_copy = zero_bool_mat.copy()

    #Recording possible answer positions by marked_zero
    marked_zero = []
    while (True in zero_bool_mat_copy):
        min_zero_row(zero_bool_mat_copy, marked_zero)
    
    #Recording the row and column positions seperately.
    marked_zero_row = []
    marked_zero_col = []
    for i in range(len(marked_zero)):
        marked_zero_row.append(marked_zero[i][0])
        marked_zero_col.append(marked_zero[i][1])

    #Step 2-2-1
    non_marked_row = list(set(range(cur_mat.shape[0])) - set(marked_zero_row))
    
    marked_cols = []
    check_switch = True
    while check_switch:
        check_switch = False
        for i in range(len(non_marked_row)):
            row_array = zero_bool_mat[non_marked_row[i], :]
            for j in range(row_array.shape[0]):
                #Step 2-2-2
                if row_array[j] == True and j not in marked_cols:
                    #Step 2-2-3
                    marked_cols.append(j)
                    check_switch = True

        for row_num, col_num in marked_zero:
            #Step 2-2-4
            if row_num not in non_marked_row and col_num in marked_cols:
                #Step 2-2-5
                non_marked_row.append(row_num)
                check_switch = True
    #Step 2-2-6
    marked_rows = list(set(range(cost_matrix.shape[0])) - set(non_marked_row))

    return(marked_zero, marked_rows, marked_cols)

def adjust_matrix(mat, cover_rows, cover_cols):
    cur_mat = mat
    non_zero_element = []

    #Step 4-1
    for row in range(len(cur_mat)):
        if row not in cover_rows:
            for i in range(len(cur_mat[row])):
                if i not in cover_cols:
                    non_zero_element.append(cur_mat[row][i])
    if len(non_zero_element)>0:                
        min_num = min(non_zero_element)

    #Step 4-2
    for row in range(len(cur_mat)):
        if row not in cover_rows:
            for i in range(len(cur_mat[row])):
                if i not in cover_cols:
                    cur_mat[row, i] = cur_mat[row, i] - min_num
    #Step 4-3
    for row in range(len(cover_rows)):  
        for col in range(len(cover_cols)):
            cur_mat[cover_rows[row], cover_cols[col]] = cur_mat[cover_rows[row], cover_cols[col]] + min_num
    return cur_mat

def isFinalResult(mat:np.ndarray):
    rows_zeros = np.count_nonzero(mat, axis=1)
    cols_zeros = np.count_nonzero(mat.T, axis=1)
    return np.all(rows_zeros>=mat.shape[1]-1) and np.all(cols_zeros>=mat.shape[0]-1)

def get_matches(mat:np.ndarray):
    pairs = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i,j]==0:
                pairs.append((i,j))
                break
    
    return pairs

def hungarian_algorithm(cost_matrix): 
    dim = min(cost_matrix.shape[0], cost_matrix.shape[1])
    if(dim == 0):
        return []
    cur_mat = cost_matrix.copy()

    #Step 1 - Every column and every row subtract its internal minimum
    for row_num in range(cost_matrix.shape[0]): 
        cur_mat[row_num] = cur_mat[row_num] - np.min(cur_mat[row_num])

    if isFinalResult(cur_mat):
        return get_matches(cur_mat)
    
    for col_num in range(cost_matrix.shape[1]): 
        cur_mat[:,col_num] = cur_mat[:,col_num] - np.min(cur_mat[:,col_num])

    if isFinalResult(cur_mat):
        return get_matches(cur_mat)
    
    zero_count = 0
    ans_pos=[]
    while zero_count < dim:
        #Step 2 & 3
        ans_pos, marked_rows, marked_cols = mark_matrix(cost_matrix, cur_mat)
        zero_count = len(marked_rows) + len(marked_cols)

        if zero_count < dim:
            cur_mat = adjust_matrix(cur_mat, marked_rows, marked_cols)

    return ans_pos

def ans_calculation(mat, pos):
    total = 0
    ans_mat = np.zeros((mat.shape[0], mat.shape[1]))
    for i in range(len(pos)):
        total += mat[pos[i][0], pos[i][1]]
        ans_mat[pos[i][0], pos[i][1]] = mat[pos[i][0], pos[i][1]]
    return total, ans_mat

def main():

    '''Hungarian Algorithm: 
    Finding the minimum value in linear assignment problem.
    Therefore, we can find the minimum value set in net matrix 
    by using Hungarian Algorithm. In other words, the maximum value
    and elements set in cost matrix are available.'''

    #The matrix who you want to find the minimum sum
    #cost_matrix = np.array([[7, 6, 2, 9, 2, 1],
    #            [6, 2, 1, 3, 9, 10],
    #            [9, 6, 8, 9, 5, 4],
    #            [6, 8, 5, 8, 6, 3],
    #            [9, 5, 6, 4, 7, 2]])

    cost_matrix = np.array([[24.33105012119288, 18.43908891458577, 2.23606797749979, 200.8482013860219, 43.41658669218482, 253.1402773167478, 176.9180601295413, 253.1402773167478, 6.324555320336759, 255.8124312851117],
                            [25.07987240796891, 22.82542442102665, 7.615773105863909, 206.4000968992021, 47.70744176750625, 252.7864711569826, 182.6499384067785, 252.7864711569826, 1, 255.9316314956008],
                            [226.1061697521764, 235.3720459187964, 252.0019841191732, 230.2172886644268, 217.0829334609241, 41.23105625617661, 237.065391822594, 41.23105625617661, 250, 22.3606797749979],
                            [28.0713376952364, 20.591260281974, 39.20459156782532, 164.7543626129518, 5.385164807134504, 230.13908837918, 142.351677194194, 230.13908837918, 42.94182110716778, 230.3128307324627],
                            [161.0093165006299, 152.9705854077835, 168.8342382338369, 31.62277660168379, 130.862523283024, 252.3885892824792, 14.14213562373095, 252.3885892824792, 174.6424919657298, 241.8677324489565],
                            [15, 30.61045573002793, 38.47076812334269, 199.6421799119615, 40.024992192379, 217.9380645963435, 179.0446871593793, 217.9380645963435, 34.0147027033899, 221.4881486671465],
                            [112.8007092176286, 113.1370849898476, 132.2308587282106, 107.7032961426901, 88.45903006477066, 174.6424919657298, 100, 174.6424919657298, 134.5362404707371, 167.6305461424021],
                            [218.0091741188889, 230.13908837918, 244.0512241313286, 267.2901045680517, 216.8155898453799, 20.591260281974, 268.9312179721796, 20.591260281974, 240.2998127340094, 39.293765408777]])

    # cost_matrix = np.array(
    #    [[0, 0, 0, 0, 0, 0, 0, 0, 1],
    #     [0, 0, 0, 0, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 0, 0],
    #     [1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 1, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 0, 0, 0, 0]]
    # )*-1

    ans_pos = hungarian_algorithm(cost_matrix.copy())#Get the element position.
    print(ans_pos)
    ans, ans_mat = ans_calculation(cost_matrix, ans_pos)#Get the minimum or maximum value and corresponding matrix.

    #Show the result
    print(f"Linear Assignment problem result: {ans:.0f}\n{np.round(ans_mat)}")

    # #If you want to find the maximum value, using the code as follows: 
    # #Using maximum value in the cost_matrix and cost_matrix to get net_matrix
    # profit_matrix = np.array([[7, 6, 2, 9],
    #             [6, 2, 1, 3],
    #             [5, 6, 8, 9],
    #             [6, 8, 5, 8],
    #             [9, 5, 6, 4]])
    # max_value = np.max(profit_matrix)
    # cost_matrix = profit_matrix
    # ans_pos = hungarian_algorithm(cost_matrix.copy())#Get the element position.
    # print(ans_pos)
    # ans, ans_mat = ans_calculation(profit_matrix, ans_pos)#Get the minimum or maximum value and corresponding matrix.
    # #Show the result
    # print(f"Linear Assignment problem result: {ans:.0f}\n{ans_mat}")

if __name__ == '__main__':
    main()