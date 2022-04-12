from collections import namedtuple
import numpy as np
from scipy.sparse.linalg import eigsh

block=namedtuple("block",["size","basis_size","op_dict"])
enlargedblock=namedtuple("enlargedblock",["size","basis_size","op_dict"])

Sz=np.array([[0.5,0],[0,-0.5]])
Sp=np.array([[0,1],[0,0]])
H=np.zeros([2,2])

initialblock=block(1,2,{"h":H,"sz":Sz,"sp":Sp})

def enlarge_block(block):
    operator=block.op_dict
    enlarged_dict={}
    enlarged_dict["h"]=np.kron(operator["h"],np.eye(2))+np.kron(operator["sz"],Sz) \
                      +0.5*np.kron(operator["sp"],Sp.conjugate().transpose()) \
                      +0.5*np.kron(operator["sp"].conjugate().transpose(),Sp)
    enlarged_dict["sz"]=np.kron(np.eye(block.basis_size),Sz)
    enlarged_dict["sp"]=np.kron(np.eye(block.basis_size),Sp)
    
    enlarged_block=enlargedblock((block.size+1),(block.basis_size*2),enlarged_dict)
    
    return enlarged_block

def truncate(op, tmatrix):
    transformed_op=tmatrix.conjugate().transpose().dot(op.dot(tmatrix))
    return transformed_op   

def single_dmrg_step(system,environment,m):
    
    sys_enlarged=enlarge_block(system)
    
    if system is environment:
        env_enlarged=sys_enlarged
    else :
        env_enlarged=enlarge_block(environment)
    
    sys_enlarged_basis=sys_enlarged.basis_size
    env_enlarged_basis=env_enlarged.basis_size
    
    sys_enlarged_op=sys_enlarged.op_dict
    env_enlarged_op=env_enlarged.op_dict
    
    H_superblock=np.kron(sys_enlarged_op["h"],np.eye(env_enlarged_basis))\
                  +np.kron(np.eye(sys_enlarged_basis),env_enlarged_op["h"])\
                  +np.kron(sys_enlarged_op["sz"],env_enlarged_op["sz"]) \
            +0.5*np.kron(sys_enlarged_op["sp"],env_enlarged_op["sp"].conjugate().transpose())\
            +0.5*np.kron(sys_enlarged_op["sp"].conjugate().transpose(),env_enlarged_op["sp"])
    
    (gs_energy,),gs = eigsh(H_superblock, k=1, which="SA")
    psi=gs.reshape([sys_enlarged_basis, -1], order="C")

    d=H_superblock.shape[0]
    rho=np.dot(gs.reshape(d,1),gs.reshape(1,d))
    cor=np.kron(sys_enlarged_op["sz"],np.eye(env_enlarged_basis))
    measure=np.trace(np.dot(rho,cor))
    
    rho=np.dot(psi, psi.conjugate().transpose())
    evals, evecs =np.linalg.eigh(rho)
    
    possible_eigenstates = []
    for eva, evec in zip(evals, evecs.transpose()):
        possible_eigenstates.append((eva, evec))
    possible_eigenstates.sort(reverse=True, key=lambda x: x[0])  # largest eigenvalue first
    
    my_m = min(len(possible_eigenstates), m)
    T = np.zeros((sys_enlarged_basis, my_m), dtype='d', order='F')
    for i, (eval, evec) in enumerate(possible_eigenstates[:my_m]):
        T[:, i] = evec

    truncation_error = 1 - sum([x[0] for x in possible_eigenstates[:my_m]])
    
    truncated_operator={}
    truncated_operator["h"]=truncate(sys_enlarged_op["h"],T)
    truncated_operator["sz"]=truncate(sys_enlarged_op["sz"],T)
    truncated_operator["sp"]=truncate(sys_enlarged_op["sp"],T)
    Block=namedtuple("Block",["size","basis_size","op_dict"])
    newblock =Block(size=sys_enlarged.size,basis_size=my_m,op_dict=truncated_operator)
    
    return newblock,gs_energy



block_store={}

Block=initialblock
block_store["l", Block.size] = Block
block_store["r", Block.size] = Block

L=20
m=20
measure=[]

while 2*Block.size < L:
    Block,energy=single_dmrg_step(Block,Block,m)
    
    block_store["l", Block.size] = Block
    block_store["r", Block.size] = Block
    
sys="l"
env="r"

sys_block=Block

energy=0
while True:
        env_block=block_store[env,L-sys_block.size-2]
        
        counter=0
    
        if  env_block.size == 1:
            sys,env=env,sys
            sys_block,env_block=env_block,sys_block
            
        
        sys_block, energy=single_dmrg_step(sys_block, env_block, m=m)
            
        block_store[sys, sys_block.size] = sys_block
        
        if sys == "l" and 2 * sys_block.size == L:
            break 
            
print(energy)   
import numpy as np
import quimb as qu

def Hq(i):
    return(qu.ham_heis(i,j=1,b=0,cyclic=False,sparse=True))

qu.groundenergy(Hq(20))
#QUIMB
from quimb import *
from quimb.tensor import *
H = MPO_ham_heis(20, cyclic=False)
dmrg=DMRG2(H)
dmrg.solve(tol=1e-6, verbosity=1) 
#TENPY
import numpy as np
import tenpy
from tenpy.networks.mps import MPS
from tenpy.models.spins import SpinChain
from tenpy.algorithms import dmrg

model_params = {
    'L': L,
    'J':1,
    'conserve': None,
}
M = tenpy.models.spins.SpinChain(model_params)
psi = MPS.from_lat_product_state(M.lat, [['up']])
dmrg_params = {
    'mixer': True, 
    'max_E_err': 1.e-10,
    'trunc_params': {
        'chi_max': 100,
        'svd_min': 1.e-10,
    },
    'verbose': True,
    'combine':True
}
eng = dmrg.TwoSiteDMRGEngine(psi,model=M, options=dmrg_params)
E, psi = eng.run() 

print(E) 

