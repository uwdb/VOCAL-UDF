from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent
import signal
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from sklearn.cluster import KMeans
import multiprocessing

def call_timeout(timeout, func, args=(), kwargs={}):
    if type(timeout) not in [int, float] or timeout <= 0.0:
        print("Invalid timeout!")

    elif not callable(func):
        print("{} is not callable!".format(type(func)))

    else:
        p = multiprocessing.Process(target=func, args=args, kwargs=kwargs)
        p.start()
        p.join(timeout)

        if p.is_alive():
            p.terminate()
            raise TimeoutError("Function call timed out")
        else:
            return True

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out")

# Set the signal handler for the SIGALRM signal
signal.signal(signal.SIGALRM, timeout_handler)

def function_with_timeout(args, func, timeout_duration=10):
    result = []
    for i, arg in enumerate(zip(*args)):
        try:
            # Set an alarm for timeout_duration seconds
            signal.alarm(timeout_duration)
            # Call the function and append the result
            result.append(func(*arg))
        except TimeoutError:
            print(f"Timeout occurred for task {i}")
            raise
        finally:
            # Disable the alarm
            signal.alarm(0)
    return result

def function_with_multithreading_and_timeout(df, func, timeout_duration=60):
    # with ThreadPoolExecutor(max_workers=8) as executor:
        # result = list(tqdm(executor.map(func, *args), total=len(df), desc="exec_udf_with_data"))
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all the tasks to the executor
        futures = [executor.submit(func, *arg) for arg in zip(*args)]

        results = []
        try:
            # Collect results with individual timeouts for each task
            for future in futures:
                result = future.result(timeout=5)
                results.append(result)
        except concurrent.futures.TimeoutError:
            # If any task times out, we can cancel all the other tasks
            for future in futures:
                future.cancel()
            raise TimeoutError("One or more function calls timed out due to exceeding the individual task timeout")
        except Exception as e:
            print("Exception occurred:", e)
        return results


def py_green(img, o0_x1, o0_y1, o0_x2, o0_y2):
    from sklearn.cluster import KMeans
    cropped_img = img[min(o0_y1, o0_y2): max(o0_y1, o0_y2), min(o0_x1, o0_x2): max(o0_x1, o0_x2)]
    reshaped_img = cropped_img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=1)
    dominant_color = kmeans.fit(reshaped_img).cluster_centers_[0]
    # time.sleep(o0_y2)
    return dominant_color[1] > dominant_color[0] and dominant_color[1] > dominant_color[2]

# Test the function with different durations
# create df with columns img, o0_x1, o0_y1, o0_x2, o0_y2, filled with random values and 20 rows. img is numpy array with shape (224, 224, 3)
np.random.seed(0)
df = pd.DataFrame({
    'img': [np.random.randint(0, 255, (224, 224, 3)) for _ in range(20)],
    'o0_x1': np.random.randint(0, 224, 20),
    'o0_y1': np.random.randint(0, 224, 20),
    'o0_x2': np.random.randint(0, 224, 20),
    'o0_y2': [i for i in range(20)]
})

_start = time.time()
args = [df['img'], df['o0_x1'], df['o0_y1'], df['o0_x2'], df['o0_y2']]
results = []
try:
    # results = function_with_timeout(args, py_green)
    results = function_with_multithreading_and_timeout(df, py_green)

except Exception as e:
    print("Exception occurred:", e)
print(results)
print("Time taken:", time.time() - _start)

# [True, True, False, True, True, False, False, True, False, False, False, True, False, False, True, False, False, True, True, False]