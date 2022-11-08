#https://websockets.readthedocs.io/en/stable/ 
#https://websockets.readthedocs.io/en/stable/intro/tutorial2.html#broadcast

import websockets
import asyncio

import configparser
import time
import sys
#----------------------------------------------------------
config = configparser.ConfigParser()
config.read('config.ini')
IP_SB = config['DEFAULT']['ip_sb']
PORT_SB = config['DEFAULT']['port_sb']
#----------------------------------------------------------


connected = set()
async def echo(websocket, path):
    connected.add(websocket)
    async for message in websocket:
        # for conn in connected:
        #     if conn!= websocket:
        #         await conn.send(message)
        print("receive msg: ", message)
        websockets.broadcast(connected, message)

start_server = websockets.serve(echo, IP_SB, PORT_SB)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
