import os
import sys
import time
import configparser

# se reciben las ip y los puertos
arguments = sys.argv
IP_TX = arguments[1]
PORT_TX = arguments[2]
PORT_RX = arguments[3]
PORT_SB = arguments[4]
CONF_MIN = arguments[5]


# se crea el archivo config.ini
config = configparser.ConfigParser()
config['DEFAULT'] = {'IP_TX': IP_TX,
		     'IP_SB': IP_TX,
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
