from smbus import SMBus
from time import sleep

addr = 8
bus = SMBus(0)
while True:
	data = [2,1]
	bus.write_i2c_block_data(addr, 0, data)
