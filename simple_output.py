import spectrometer as sp
import numpy as np

front = sp.pyEDPSolver.fromfile("/home/sgsdxzy/Sync/Programs/light-spectrometer/front_solver.npz")
print(front.getP(248,0))
