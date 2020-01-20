import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl
import matplotlib.pyplot as plt
import numpy as np


# Set parameters and load dataset
max_iter = 30
max_cg_iter = 5
lamda = 0.001

ksp_file = 'data/liver/ksp.npy'
coord_file = 'data/liver/coord.npy'

device = sp.Device(-1)

xp = device.xp
device.use()

# Load datasets.
ksp = xp.load(ksp_file)
coord = xp.load(coord_file)

print(f'K-space shape: {ksp.shape}')
print(f'K-space dtype: {ksp.dtype}')
print(f'K-space (min, max): ({np.abs(ksp).min()}, {np.abs(ksp).max()})')
print(f'Coord shape: {coord.shape}')  # (na, ns, 2)
print(f'Coord shape: {coord.dtype}')
print(f'Coord (min, max): ({coord.min()}, {coord.max()})')

plt.ion()
f, ax = plt.subplots(1, 1)
ax.scatter(coord[:15, :, -1], coord[:15, :, -2])

# Use JSENSE to estimate sensitivity maps
mps = mr.app.JsenseRecon(ksp, coord=coord, device=device).run()

print(f'Shape of coil sensitivity maps: {mps.shape}')

pl.ImagePlot(mps)

# Primal dual hybrid gradient reconstruction
pdhg_app = mr.app.TotalVariationRecon(ksp, mps, lamda=lamda, coord=coord,
                                      max_iter=max_iter,
                                      device=device,
                                      save_objective_values=True)
print(f'Name of solver: {pdhg_app.alg_name}')
pdhg_img = pdhg_app.run()

print(f'Image shape: {pdhg_img.shape}')
print(f'Image dtype: {pdhg_img.dtype}')

pl.ImagePlot(pdhg_img)

# PDHG with dcf
# Compute preconditioner
precond_dcf = mr.pipe_menon_dcf(coord, device=device)

print(f'DCF shape: {precond_dcf.shape}')
print(f'DCF dtype: {precond_dcf.dtype}')

f, ax = plt.subplots(1, 1)
ax.imshow(precond_dcf)

precond_dcf = xp.tile(precond_dcf, [len(mps)] + [1] * (mps.ndim - 1))
img_shape = mps.shape[1:]
G = sp.linop.FiniteDifference(img_shape)
max_eig_G = sp.app.MaxEig(G.H * G).run()
sigma2 = xp.ones([sp.prod(img_shape) * len(img_shape)],
                 dtype=ksp.dtype) / max_eig_G
sigma = xp.concatenate([precond_dcf.ravel(), sigma2.ravel()])

pdhg_dcf_app = mr.app.TotalVariationRecon(ksp, mps, lamda=lamda, coord=coord,
                                          sigma=sigma, max_iter=max_iter,
                                          device=device,
                                          save_objective_values=True)
print(f'Name of solver: {pdhg_dcf_app.alg_name}')
pdhg_dcf_img = pdhg_dcf_app.run()

pl.ImagePlot(pdhg_dcf_img)

# PDHG with single-channel k-space preconditioning
# Compute preconditioner
ones = np.ones_like(mps)
ones /= len(mps)**0.5
precond_sc = mr.kspace_precond(ones, coord=coord, device=device)

print(f'Shape of k-space precond: {precond_sc.shape}')
print(f'Dtype of k-space precond: {precond_sc.dtype}')

pl.ImagePlot(precond_sc)

img_shape = mps.shape[1:]
max_eig_G = sp.app.MaxEig(G.H * G).run()
sigma2 = xp.ones([sp.prod(img_shape) * len(img_shape)],
                 dtype=ksp.dtype) / max_eig_G
sigma = xp.concatenate([precond_sc.ravel(), sigma2.ravel()]) / 2

pdhg_sc_app = mr.app.TotalVariationRecon(ksp, mps, lamda=lamda, coord=coord,
                                         sigma=sigma, max_iter=max_iter,
                                         device=device,
                                         save_objective_values=True)
print(f'Name of solver: {pdhg_sc_app.alg_name}')
pdhg_sc_img = pdhg_sc_app.run()

pl.ImagePlot(pdhg_sc_img)

# PDHG with multi-channel k-space preconditioning
# Compute preconditioner
precond_mc = mr.kspace_precond(mps, coord=coord, device=device)

print(f'Shape of k-space precond: {precond_mc.shape}')
print(f'Dtype of k-space precond: {precond_mc.dtype}')

pl.ImagePlot(precond_mc)

img_shape = mps.shape[1:]
max_eig_G = sp.app.MaxEig(G.H * G).run()
sigma2 = xp.ones([sp.prod(img_shape) * len(img_shape)],
                 dtype=ksp.dtype) / max_eig_G
sigma = xp.concatenate([precond_mc.ravel(), sigma2.ravel()])

pdhg_mc_app = mr.app.TotalVariationRecon(ksp, mps, lamda=lamda, coord=coord,
                                         sigma=sigma, max_iter=max_iter,
                                         device=device,
                                         save_objective_values=True)
print(f'Name of solver: {pdhg_mc_app.alg_name}')
pdhg_mc_img = pdhg_mc_app.run()

pl.ImagePlot(pdhg_mc_img)

# Plot convergence curves
plt.figure(figsize=(8, 3))
plt.semilogy(pdhg_app.objective_values,
             marker='+', color='C3')
plt.semilogy(pdhg_dcf_app.objective_values,
             marker='s', color='C4')
plt.semilogy(pdhg_sc_app.objective_values,
             marker='*', color='C5')
plt.semilogy(pdhg_mc_app.objective_values,
             marker='x', color='C6')
plt.legend(['PDHG',
            'PDHG w/ density comp.',
            'PDHG w/ SC k-space precond.',
            'PDHG w/ MC k-space precond.'])
plt.ylabel('Objective Value [a.u.]')
plt.xlabel('Iteration Number')
plt.title(r"Total Variation Regularized Reconstruction")
plt.tight_layout()
