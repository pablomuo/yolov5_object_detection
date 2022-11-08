import os
import sys
import time
import configparser

# se reciben las ip y los puertos
arguments = sys.argv
IP_TX = arguments[1]
IP_SB = arguments[2]
PORT_TX = arguments[3]
PORT_RX = arguments[4]
PORT_SB = arguments[5]
CONF_MIN = arguments[6]


# se crea el archivo config.ini
config = configparser.ConfigParser()
config['DEFAULT'] = {'IP_TX': IP_TX,
		     'IP_SB': IP_SB,
                     'PORT_TX': PORT_TX,
                     'PORT_RX': PORT_RX,
		     'PORT_SB': PORT_SB,
		     'CONF_MIN': CONF_MIN}
with open('config.ini', 'w') as configfile:
  config.write(configfile)

# se ejecutan los códigos
os.system('python servidor_broadcast.py &')

time.sleep(3)

os.system('python detect.py --source 0 --weights best.pt')
