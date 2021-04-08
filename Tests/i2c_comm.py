from smbus import SMBus
from time import sleep

#initialize i2c bus
addr = 8
bus = SMBus(0)

#send mock data through i2c
while True:
	data = [0, 0]
	bus.write_i2c_block_data(addr, 0, data)
