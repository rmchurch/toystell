from toystell import ToyStellarator
import time

tsObj = ToyStellarator(R0=10.0, a=1.5, B_t=5.0, iota_two_thirds=0.9, f_ren=1.0, \
                       T0=5.0, n0=2.0, alphaT=2.0, alphan=2.0, Z_eff=1.0)

print("Start stationatry_plasma_point")
tstart = time.time()
tsObj.stationary_plasma_point(tsObj.fparams)
print("End stationary_plasma_point %0.2f sec" % (time.time() - tstart))

print("Start plot_Bscan")
tsObj.T0, tsObj.n0 = 5.0, 1.0
tsObj.R0, tsObj.a, tsObj.B_t = 6.0, 6.0/5.5, 3.0
tsObj.plot_Bscan()
print("End plot_Bscan %0.2f sec" % (time.time() - tstart))

#TODO: Currently taking too long, something not right
print("Start plot_nTscan")
#tsObj.R0, tsObj.a, tsObj.B_t = 7.0, 7.0/8.0, 2.0
tsObj.R0, tsObj.a, tsObj.B_t = 4.0, 4.0/5.0, 2.5
tsObj.plot_nTscan()
print(tsObj.fparams)
print("End plot_nTscan %0.2f sec" % (time.time() - tstart))