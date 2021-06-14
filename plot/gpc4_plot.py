import numpy as np
import matplotlib.pyplot as plt


def read_file(filename):
    pure_data = []
    wreader_data = []
    threads_data = []
    with open(filename,'r') as f:
        data = f.readlines()
        for line in data:
            sp = line.split(':')
            pure = sp[1].split()[0]
            wreader = sp[2].split()[0]
            threads = sp[-1].split('\n')[0]
            pure_data.append(int(pure))
            wreader_data.append(int(wreader))
            threads_data.append(int(threads))
    return pure_data, wreader_data, threads_data


pure_data, wreader_data, threads_data = read_file('time_cuda.data')
pure_sq_data, wreader_sq_data, _ = read_file('time_seq.data')
print(pure_data)
print(threads_data)


plt.figure(0)
plt.ylabel(r"Execution time [$\mu s$]")
plt.xlabel("Stream size") #not completly accurate in the none time() case
plt.title("time()")
plt.yscale('log')
plt.xscale('log')
plt.plot(threads_data,pure_data,'r')
plt.plot(threads_data,wreader_data,'b')
plt.plot(threads_data,pure_sq_data,'g')
plt.plot(threads_data,wreader_sq_data,'y')
plt.plot(threads_data,pure_data,'rx')
plt.plot(threads_data,wreader_data,'bx')
plt.plot(threads_data,pure_sq_data,'gx')
plt.plot(threads_data,wreader_sq_data,'yx')
plt.legend(["cuda only","cuda with Reader","seq only","seq with Reader"])

plt.show()