from multiprocessing import Pool
import time

def new_awesome_function(a,b):
    print(a,b, 'start')
    time.sleep(1)
    print(a,b, 'end')
    return a + b

if __name__ == '__main__':
    data = [1,2,3,4,5]
    pool = Pool(processes=4)
    results = []
    for i, x in enumerate(data):
        r = pool.apply_async(new_awesome_function, (i, x))
        results.append((i,r))
    pool.close()
    already = []
    while len(already) < len(data):
        for i,r in results:
            if r.ready() and i not in already:
                already.append(i)
                print(i, 'is ready!')
    pool.join()
