from gpaw.response.df import DielectricFunction
from gpaw.response.qeh import BuildingBlock

df = DielectricFunction(calc='MoS2_gs_fulldiag.gpw',
                        eta=0.001,
                        domega0=0.05,
                        omega2=10.0,
                        nblocks=8,
                        ecut=150,
                        name='MoS2_response_',
                        truncation='2D')

buildingblock = BuildingBlock('MoS2', df)

buildingblock.calculate_building_block()
