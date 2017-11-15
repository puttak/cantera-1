import time
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
import numpy as np

# A function that can be called to do work:
def work(arg,df):
    cpu_id =int(multiprocessing.current_process().name.split('-')[1])*1000
    print(cpu_id)

    print("Function receives the arguments as a list:", arg)
    # Split the list to individual variables:
    i, j = arg
    state = np.hstack([i, j])
    df.loc[cpu_id]=state
    # All this work function does is wait 1 second...
    time.sleep(1)
    # ... and prints a string containing the inputs:
    print("%s_%s" % (i, j))

    return "%s_%s" % (i, j)


# List of arguments to pass to work():
arg_instances = [(1, 1), (1, 2), (1, 3), (1, 4)
#                 (1, 1), (2, 1), (3, 1), (4, 1)
                 ]
columnNames = ['i','j']
df = pd.DataFrame(columns=columnNames)

# Anything returned by work() can be stored:
# results = Parallel(n_jobs=4, verbose=1, backend="threading")(map(delayed(work), arg_instances))
results = Parallel(n_jobs=4, verbose=1)(delayed(work)(arg_instances,df))
print(results)
