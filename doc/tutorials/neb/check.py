from ase.io import read

ets = read('neb2.traj').get_potential_energy()
ef = read('final.traj').get_potential_energy()
print(ets - ef)
assert abs(ets - ef - 0.295) < 0.003
