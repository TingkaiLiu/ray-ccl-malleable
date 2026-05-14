#!/usr/bin/env python3
import sys
from mpi4py import MPI
import numpy
import kernels
import time
import argparse
import json

COMM_TIME, COMP_TIME, ITER_TIME = range(3)

def factor(r):
    fac1 = int(numpy.sqrt(r+1.0))
    fac2 = 0
    for fac1 in range(fac1, 0, -1):
        if r%fac1 == 0:
            fac2 = int(r/fac1) 
            break
    return fac1, fac2

def output_timing(data, n, m, np, iterations, warmup, nodetype, nodenum):
    from datetime import datetime
    import os
    import subprocess
    
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    
    jobid = os.environ.get("SLURM_JOB_ID", "unknown")
    nodelist = "unknown"
    
    try:
        cmd = f"squeue -j {jobid} -h -o %N"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            nodelist = result.stdout.strip()
            if '[' in nodelist and ']' in nodelist:
                import re
                nodes_inside_brackets = re.search(r'\[(.*?)\]', nodelist).group(1)
                nodes_without_commas = nodes_inside_brackets.replace(',', '')
                prefix = nodelist.split('[')[0]
                nodelist = prefix + nodes_without_commas
    except Exception as e:
        print(f"Error getting nodelist: {e}")
    
    # Create output directory if it doesn't exist
    os.makedirs("stencil_sync_results", exist_ok=True)
    
    # Create the detailed timestamp-based filename (for header)
    detailed_filename = f"{dt_string}_node{nodelist}_n{n}_m{m}_np{np}_i{iterations}_w{warmup}_mpi_{nodetype}"
    
    # Create the experiment type identifier and CSV filename
    experiment_type = f"nodenum{nodenum}_n{n}_m{m}_np{np}_i{iterations}_w{warmup}_{nodetype}"
    output_filename = f"stencil_sync_results/{experiment_type}.csv"
    
    header = "Total Time,Communication Time,Computation Time"
    
    combined_timing = numpy.zeros((iterations, 3))
    
    for process_data in data:
        combined_timing += process_data[warmup:]
    
    final_avg = numpy.sum(combined_timing, axis=0)
    
    final_avg = (final_avg / 1e9) / np
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(output_filename)
    
    # Modify output file format
    with open(output_filename, 'a' if file_exists else 'w') as output_file:
        if not file_exists:
            output_file.write(header + '\n')
        
        # Write detailed filename as a separate line
        output_file.write(detailed_filename + '\n')
        
        # Write numerical values as a separate line
        output_file.write(','.join([
            str(final_avg[ITER_TIME]),
            str(final_avg[COMM_TIME]),
            str(final_avg[COMP_TIME])
        ]) + '\n\n\n')
    
    print(f"Data appended to {output_filename}")

def output_profile_json(iteration_time, comm_time, comp_time):
    with open("profile.json", "w") as output_file:
        json.dump(
            {
                "execution_time": iteration_time,
                "comm_time": comm_time,
                "comp_time": comp_time,
            },
            output_file,
        )

def main():
    parser = argparse.ArgumentParser(description='Python MPI/Numpy Stencil execution on 2D grid')
    parser.add_argument('iterations', type=int, help='Number of iterations')
    parser.add_argument('n', type=int, help='Array dimension X')
    parser.add_argument('m', type=int, nargs='?', help='Array dimension Y (if not provided, uses n)')
    parser.add_argument('--nodetype', '-nt', type=str, choices=['c', 'i'], required=True, help='Node type (c or i)')
    parser.add_argument('--warmup', '-w', type=int, default=10, help='Number of warmup iterations')
    parser.add_argument('--nodenumber', '-nn', type=int, required=True, help='Node number')
    args = parser.parse_args()
    
    iterations = args.iterations
    n = args.n
    m = args.m if args.m is not None else n
    nodetype = args.nodetype
    nodenum = args.nodenumber
    
    if iterations < 1:
        sys.exit("ERROR: iterations must be >= 1")
    
    comm = MPI.COMM_WORLD
    me = comm.Get_rank() #My ID
    np = comm.Get_size() #Number of processor, NOT numpy
    x, y = factor(np)
    comm = comm.Create_cart([x,y])
    coords = comm.Get_coords(me)
    X = coords[0]
    Y = coords[1]

    x = int(x)
    y = int(y)

    nsquare = n * m

    if nsquare < np:
        sys.exit(f"ERROR: grid size {nsquare} must be at least # ranks: {np}")
    if n % x:
        sys.exit(f"ERROR: grid size {n} does not evenly divide the number of chares in the x dimension {x}")
    if m % y:
        sys.exit(f"ERROR: grid size {m} does not evenly divide the number of chares in the y dimension {y}")

    warmup=args.warmup
    if me == 0:
        print('Number of processors        = ', np)
        print('Grid shape (x,y)            = ', x, y)
        print('Problem Domain (x,y)        = ', n, m)
        print('Number of warmup iterations = ', warmup)
        print('Number of iterations        = ', iterations)
        pass

    my_blocksize = (n//x)*(m//y)
    ghost_size = ((n//x) * 2) + ((m//y) * 2) + 4
    kernels.set_block_params(m//y, n//x)

    width = m//y
    height = n//x

    T = numpy.ones(my_blocksize + ghost_size, dtype=numpy.float64)
    newT = numpy.ones(my_blocksize + ghost_size, dtype=numpy.float64)

    kernels.enforce_BC(T)

    top_buf_out = numpy.zeros(width)
    top_buf_in = numpy.zeros(width)
    bot_buf_out = numpy.zeros(width)
    bot_buf_in = numpy.zeros(width)

    right_buf_out = numpy.zeros(height)
    right_buf_in = numpy.zeros(height)
    left_buf_out = numpy.zeros(height)
    left_buf_in = numpy.zeros(height)

    if Y < y-1:
        top_nbr = comm.Get_cart_rank([X,Y+1])
    if Y > 0:
        bot_nbr = comm.Get_cart_rank([X,Y-1])
    if X > 0:
        left_nbr = comm.Get_cart_rank([X-1,Y])
    if X < x-1:
        right_nbr = comm.Get_cart_rank([X+1,Y])
    timing_info = numpy.zeros((warmup + iterations, 3), dtype=numpy.float64)
    iteration_time = {}
    comm_time_profile = {}
    comp_time_profile = {}

    for i in range(warmup + iterations):
        if me == 0:
            print("Iter:", i, flush=True)
        iter_start = time.perf_counter_ns()
        comm.Barrier()
        # start = time.perf_counter_ns()
        if Y % 2 == 0:
            if Y < y-1:
                kernels.pack_top(T, top_buf_out)
                comm.Send(top_buf_out, dest=top_nbr, tag=99)
                
            if Y > 0:
                kernels.pack_bottom(T, bot_buf_out)
                comm.Send(bot_buf_out, dest=bot_nbr, tag=101)

            if Y < y-1:
                comm.Recv(top_buf_in, source=top_nbr, tag=101)
                kernels.unpack_top(T, top_buf_in)
                
            if Y > 0:
                comm.Recv(bot_buf_in, source=bot_nbr, tag=99)
                kernels.unpack_bottom(T, bot_buf_in)

        else:
            if Y < y-1:
                comm.Recv(top_buf_in, source=top_nbr, tag=101)
                kernels.unpack_top(T, top_buf_in)
                
            if Y > 0:
                comm.Recv(bot_buf_in, source=bot_nbr, tag=99)
                kernels.unpack_bottom(T, bot_buf_in)

            if Y < y-1:
                kernels.pack_top(T, top_buf_out)
                comm.Send(top_buf_out, dest=top_nbr, tag=99)
                
            if Y > 0:
                kernels.pack_bottom(T, bot_buf_out)
                comm.Send(bot_buf_out, dest=bot_nbr, tag=101)
        
        if X % 2 == 0:
            if X < x - 1:
                kernels.pack_right(T, right_buf_out)
                comm.Send(right_buf_out, dest=right_nbr, tag=990)
                
            if X > 0:
                kernels.pack_left(T, left_buf_out)
                comm.Send(left_buf_out, dest=left_nbr, tag=1010)

            if X < x - 1:
                comm.Recv(right_buf_in, source=right_nbr, tag=1010)
                kernels.unpack_right(T, right_buf_in)
                
            if X > 0:
                comm.Recv(left_buf_in, source=left_nbr, tag=990)
                kernels.unpack_left(T, left_buf_in)

        else:
            if X < x - 1:
                comm.Recv(right_buf_in, source=right_nbr, tag=1010)
                kernels.unpack_right(T, right_buf_in)
                
            if X > 0:
                comm.Recv(left_buf_in, source=left_nbr, tag=990)
                kernels.unpack_left(T, left_buf_in)

            if X < x - 1:
                kernels.pack_right(T, right_buf_out)
                comm.Send(right_buf_out, dest=right_nbr, tag=990)
                
            if X > 0:
                kernels.pack_left(T, left_buf_out)
                comm.Send(left_buf_out, dest=left_nbr, tag=1010)
        # end = time.perf_counter_ns()
        # print(end-start, flush=True)
        comm.Barrier()
        comm_end = time.perf_counter_ns()
        comm_time = comm_end - iter_start

        # start = time.perf_counter()
        comp_start = time.perf_counter_ns()
        kernels.compute(newT, T)
        newT, T = T, newT
        kernels.enforce_BC(T)
        comp_end = time.perf_counter_ns()
        comp_time = comp_end - comp_start
        # end = time.perf_counter()
        # print(end - start, flush=True)

        iter_end = time.perf_counter_ns()

        timing_info[i][COMM_TIME] = comm_time
        timing_info[i][ITER_TIME] = iter_end - iter_start
        timing_info[i][COMP_TIME] = comp_time
        if me == 0:
            iteration_time[i + 1] = timing_info[i][ITER_TIME] / 1e9
            comm_time_profile[i + 1] = timing_info[i][COMM_TIME] / 1e9
            comp_time_profile[i + 1] = timing_info[i][COMP_TIME] / 1e9
    data = comm.gather(timing_info, root=0)
    if me == 0:
        output_timing(data, n, m, np, iterations, warmup, nodetype, nodenum)
        output_profile_json(iteration_time, comm_time_profile, comp_time_profile)

if __name__ == '__main__':
    main()