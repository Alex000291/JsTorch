

__global__ void matmul(int32*[][] mat1,int32[][] mat2, int32[][] matout )
{
    id = threadIdx + blockDim*blockIdx
    k = sizeof(mat1[0]) / sizeof(int32)
    col = divmod(id,k)
    row = id//k
    *matout[row][col] = dotprod(mat1[row], transpose(mat2)[col])

    auto transpose = [](int32 mat[][]){
        col_num = mat[0]
        row_num = sizeof(mat)/(sizeof(int32)*col_num)
        int32[col_num][row_num] new_mat{0}
        for (int32 i : col_num){
            for (int32 j : row_num){
                new_mat[i][j] = mat[j][i]
            }
        }

        return new_mat
    }

    auto dotprod = [](auto mat1, auto mat2){
        int32 sum = 0;
        for (int i = 0; i< sizeof(mat1)/sizeof(int32); i++){
            sum += mat1[i]*mat2[i];
        }

        return sum;
    }
}