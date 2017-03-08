import spectrometer as sp
import numpy as np

#B_data = np.loadtxt("/home/sgsdxzy/Sync/Programs/light-spectrometer/magnet2.txt", skiprows=4)
spec = sp.pySpectrometer.fromfile("/home/sgsdxzy/Sync/Programs/light-spectrometer/spectrometer1.npz")
#spec.init(B_data, x_delta=0.002, y_delta=0.0025, x_offset=140e-3, y_offset=-110e-3, dt_multiplier=0.0001, maxtime=1e-4)
side, front = spec.getSolvers(np.arange(30, 400, 0.1, dtype=float), np.arange(-10,11,1, dtype=float))
if side is not None :
    side.save("/home/sgsdxzy/Sync/Programs/light-spectrometer/side_solver.npz")
if front is not None:
    front.save("/home/sgsdxzy/Sync/Programs/light-spectrometer/front_solver.npz")
#print(solvers.positions.shape)
#print(front.getP(247,0))
