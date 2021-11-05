from numba import vectorize, int64
import numpy as np
import time

num_loops = 500
img1 = np.ones((1000, 1000), np.int64) * 5
img2 = np.ones((1000, 1000), np.int64) * 10
img3 = np.ones((1000, 1000), np.int64) * 15


@vectorize([int64(int64,int64,int64)], target='cuda')
def add_arrays_numba(img1, img2, img3):
    return np.square(img1+img2+img3)

start2 = time.time()
for i in range(num_loops):
    result = add_arrays_numba(img1, img2, img3)
end2 = time.time()
run_time2 = end2 - start2
print('Total time using numba accelerating={}'.format(run_time2))
print('Average time using numba accelerating={}'.format(run_time2/num_loops))