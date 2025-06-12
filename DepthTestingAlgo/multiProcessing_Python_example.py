import multiprocessing

def calSQ(a):
    print(a * a)
    
def calCube(a):
    print(a * a * a)

t1 = multiprocessing.Process(target=calSQ, args=(20,))
t2 = multiprocessing.Process(target=calCube, args=(10,))

t1.start()
t2.start()

t1.join()
t2.join()

