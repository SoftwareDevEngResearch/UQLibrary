# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 18:10:02 2022

@author: USER
"""

import numpy as np
import mpi4py.MPI as MPI


mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()
eval_fcn = lambda pois : np.array([np.sum(pois, axis =1), -np.sum(pois, axis = 1)]).transpose()
poi_sample = np. array([[1, 0], [1, -1], [2, -1], [2, -2], [3, -2]],dtype = float)
logging=2
if mpi_rank == 0:
    print("poi_sample in thread " + str(mpi_rank) + ": " + str(poi_sample))
    for i_rank in range(mpi_size):
        if mpi_rank == 0:
            samp_per_subsample = int(np.floor(poi_sample.shape[0]/mpi_size))
            if i_rank == 0:
                data = poi_sample[0:samp_per_subsample]
            else: 
                if i_rank == mpi_size-1:
                    data_broadcast = poi_sample[(i_rank*samp_per_subsample):]
                else:
                    data_broadcast = poi_sample[(i_rank*samp_per_subsample):((i_rank+1)*samp_per_subsample)]
                mpi_comm.send(data_broadcast.shape, dest = i_rank, tag = 0)
                mpi_comm.Send([data_broadcast,MPI.DOUBLE],dest = i_rank, tag = 1)
                #print("poi_subsample sent to thread " + str(i_rank) + ": " + str(data_broadcast))
else:
    data_shape = mpi_comm.recv(source = 0, tag = 0)
    data = np.empty(data_shape)
    mpi_comm.Recv(data,source=0, tag=1)
            

# Evaluate each subsamples
qoi_subsample = eval_fcn(data)
mpi_comm.Barrier()

if mpi_rank == 0:
    qoi_sample = np.zeros((poi_sample.shape[0], qoi_subsample.shape[1]), dtype = float)
#print(poi_reconstructed)

if mpi_rank > 0:
    mpi_comm.send(qoi_subsample.shape, dest = 0, tag = 0)
    mpi_comm.Send([qoi_subsample, MPI.DOUBLE], dest = 0, tag =1)
    #print("sending data from thread " + str(mpi_rank) + ": " + str(data))
elif mpi_rank ==0 :
    total_samp=0
    for i_rank in range(mpi_size):
        if i_rank > 0:
            subsample_shape = mpi_comm.recv(source = i_rank, tag = 0)
            #print("receiving data from thread " + str(i_rank) + " of shape: " + str(data_shape))
        else :
            subsample_shape = qoi_subsample.shape
        n_samp = subsample_shape[0]
        if i_rank > 0:
            #print("poi_reconstructed before receiving: " + str(poi_reconstructed))
            mpi_comm.Recv(qoi_sample[total_samp:(total_samp+n_samp)], source = i_rank, tag=1)
        else :
            qoi_sample[total_samp:(total_samp+n_samp)] = qoi_subsample
        if logging > 1:
            print("qoi_reconstructed after receiving thread " + str(i_rank) + ": " + str(qoi_sample))
        total_samp += n_samp 

mpi_comm.Bcast([qoi_sample, MPI.DOUBLE], root = 0)
print("qoi_sample in thread " + str(mpi_rank) + ": " + str(qoi_sample))

# poi_sample = np.array([[1,-1], [2,-2], [3,-3], [4,-4], [5,-5], [6,-6]])
# print("Thread " + str(mpi_rank) + ", poi_sample: " + str(poi_sample))
 
# # Seperate poi samples into subsample for each thread
# if mpi_rank == 0:
#     samp_per_subsample = int(np.floor(poi_sample.shape[0]/mpi_size))
#     for i_rank in range(mpi_size):
#         #Seperate subsample and then broadcast
#         if i_rank == 0:
#             poi_subsample = poi_sample[0:samp_per_subsample]
#         if i_rank == mpi_size-1:
#             poi_subsample_broadcast = poi_sample[(i_rank*samp_per_subsample):]
#         else:
#             poi_subsample_broadcast = poi_sample[(i_rank*samp_per_subsample):((i_rank+1)*samp_per_subsample)]
#         # Broadcast poi_subsample to corresponding thread
#         if i_rank > 0:
#             #MPI.Comm.Send(poi_subsample_broadcast.shape, dest = i_rank, tag = i_rank)
#             #print("Sent sample shape from thread 0 to thread " + str(i_rank))
#             MPI.Comm.send([poi_subsample_broadcast, MPI.DOUBLE], dest = i_rank, tag = 0)
#             print("Sent subsample from thread 0 to thread " + str(i_rank))
# print("Thread " + str(mpi_rank) + " hitting comm barrier")
# MPI.COMM_WORLD.Barrier()
# if mpi_rank > 0:
#     MPI.Comm.Recv(poi_subsample, source= 0, tag = 0)
# print("Thread " + str(mpi_rank) + ", poi_subsample: " + str(poi_subsample))


# send_array


# print("Hello from thread " + str(mpi_rank) + " of " + str(mpi_size))
# poi_sample = np.random.rand(11,4)
# #print("poi_sample on thread " + str(mpi_rank) + ": " + str(poi_sample))
# if mpi_rank == 0:
#     print("poi_sample in thread " + str(mpi_rank) + ": " + str(poi_sample))
#     for i_rank in range(mpi_size):
#         if mpi_rank == 0:
#             samp_per_subsample = int(np.floor(poi_sample.shape[0]/mpi_size))
#             if i_rank == 0:
#                 data = poi_sample[0:samp_per_subsample]
#             else: 
#                 if i_rank == mpi_size-1:
#                     data_broadcast = poi_sample[(i_rank*samp_per_subsample):]
#                 else:
#                     data_broadcast = poi_sample[(i_rank*samp_per_subsample):((i_rank+1)*samp_per_subsample)]
#                 comm.send(data_broadcast.shape, dest = i_rank, tag = 0)
#                 comm.Send([data_broadcast,MPI.DOUBLE],dest = i_rank, tag = 1)
#                 #print("poi_subsample sent to thread " + str(i_rank) + ": " + str(data_broadcast))
# else:
#     data_shape = comm.recv(source = 0, tag = 0)
#     data = np.empty(data_shape)
#     comm.Recv(data,source=0, tag=1)
    
    
# print("Data in thread " + str(mpi_rank) + ": " + str(data))

# if mpi_rank == 0:
#     poi_reconstructed = np.zeros(poi_sample.shape, dtype = float)
# #print(poi_reconstructed)

# if mpi_rank > 0:
#     comm.send(data.shape, dest = 0, tag = 0)
#     comm.Send([data, MPI.DOUBLE], dest = 0, tag =1)
#     #print("sending data from thread " + str(mpi_rank) + ": " + str(data))
# elif mpi_rank ==0 :
#     total_samp=0
#     for i_rank in range(mpi_size):
#         if i_rank > 0:
#             data_shape = comm.recv(source = i_rank, tag = 0)
#             #print("receiving data from thread " + str(i_rank) + " of shape: " + str(data_shape))
#         else :
#             data_shape = data.shape
#         n_samp = data_shape[0]
#         if i_rank > 0:
#             #print("poi_reconstructed before receiving: " + str(poi_reconstructed))
#             comm.Recv(poi_reconstructed[total_samp:(total_samp+n_samp)], source = i_rank, tag=1)
#         else :
#             poi_reconstructed[total_samp:(total_samp+n_samp)] = data
#         print("poi_reconstructed after receiving thread " + str(i_rank) + ": " + str(poi_reconstructed))
#         total_samp += n_samp 
#     print("Reconstructed sample error: " + str(np.sum(poi_reconstructed- poi_sample)))
