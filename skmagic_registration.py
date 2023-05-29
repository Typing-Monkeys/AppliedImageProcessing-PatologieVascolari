from matplotlib import pyplot as plt
from pystackreg import StackReg
from skimage import io
# import numpy as np

# load reference and "moved" image
ref = io.imread('./P27/P2707758/BASALE_crop/BASALE_crop0000.png')
mov = io.imread('./P27/P2707758/MDC_crop/MDC_crop0000.png')

# Translational transformation
sr = StackReg(StackReg.TRANSLATION)
out_tra = sr.register_transform(ref, mov)
out_tra = out_tra / out_tra.max()  # va normalizzata per essere salvata

plt.imshow(out_tra, cmap='gray')
io.imsave('./traslation.png', out_tra, plugin='pil')
# plt.savefig('./traslation.png')
plt.show()

# Rigid Body transformation
sr = StackReg(StackReg.RIGID_BODY)
out_rot = sr.register_transform(ref, mov)
out_rot = out_rot / out_rot.max()  # va normalizzata per essere salvata

plt.imshow(out_rot, cmap='gray')
io.imsave('./rigidbody.png', out_rot, plugin='pil')
plt.show()

# Scaled Rotation transformation
sr = StackReg(StackReg.SCALED_ROTATION)
out_sca = sr.register_transform(ref, mov)
out_sca = out_sca / out_sca.max()  # va normalizzata per essere salvata

plt.imshow(out_sca, cmap='gray')
io.imsave('./scalerotate.png', out_sca, plugin='pil')
plt.show()

# Affine transformation
sr = StackReg(StackReg.AFFINE)
out_aff = sr.register_transform(ref, mov)
out_aff = out_aff / out_aff.max()  # va normalizzata per essere salvata

plt.imshow(out_aff, cmap='gray')
io.imsave('./affine.png', out_aff, plugin='pil')
plt.show()

# Bilinear transformation
sr = StackReg(StackReg.BILINEAR)
out_bil = sr.register_transform(ref, mov)
out_bil = out_bil / out_bil.max()  # va normalizzata per essere salvata

plt.imshow(out_bil, cmap='gray')
io.imsave('./bilateral.png', out_bil, plugin='pil')
plt.show()
