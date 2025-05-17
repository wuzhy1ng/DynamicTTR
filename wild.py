import asyncio
import datetime
import json
import os.path
import traceback

import aiohttp
import websockets

MAX_NUM_ADDRESSES = 1000
WILD_DATA_PATH = './wild_data'
if not os.path.exists(WILD_DATA_PATH):
    os.makedirs(WILD_DATA_PATH)
SWAP_ADDRESSES = set()
SWAP_ACTION_FILE = open('./wild_data/swap_actions', 'w', encoding='utf-8')
SWAP_LISTENER_PATH = './wild_data'


async def listen(address: str):
    path = os.path.join(SWAP_LISTENER_PATH, address)
    file = open(path, 'w', encoding='utf-8')
    conn = websockets.connect('ws://localhost:55002/api/v1/get_flow_changes?source=' + address)
    async with conn as ws:
        while True:
            data = await ws.recv()
            data = json.loads(data)
            print(datetime.datetime.now(), 'get flow change from:', address, data)
            file.write(json.dumps(data))
            file.write('\n')
            file.flush()


async def generate_swap_addresses():
    conn = websockets.connect('ws://localhost:55002/api/v1/get_swap_action')
    async with conn as ws:
        while True:
            data = await ws.recv()
            data = json.loads(data)
            if data['address'] in SWAP_ADDRESSES or len(SWAP_ADDRESSES) > MAX_NUM_ADDRESSES:
                continue
            print(datetime.datetime.now(), 'get swapper candidate:', data)

            # check the address is valuable or not
            if not await is_valuable(data['address']):
                continue
            print(datetime.datetime.now(), 'get valuable swapper:', data['address'])
            SWAP_ADDRESSES.add(data['address'])
            SWAP_ACTION_FILE.write(json.dumps(data))
            SWAP_ACTION_FILE.write('\n')
            SWAP_ACTION_FILE.flush()
            yield data['address']


async def is_valuable(address: str) -> bool:
    # check for labeled or not
    client = aiohttp.ClientSession()
    try:
        resp = await client.post(
            url='http://localhost:55000/api/v1/get_address_label',
            json={
                "jsonrpc": "2.0",
                "id": "0",
                "method": "get_address_label",
                "params": {"addresses": [address]},
            }
        )
        data = await resp.json()
        labels = data['result'][address]
        if len(labels) > 0:
            return False

        # check the address is contract or not
        resp = await client.post(
            url='https://mainnet.chainnodes.org/9c408170-0042-4ff0-bb1c-2f3e44284644',
            json={
                "jsonrpc": "2.0",
                "method": "eth_getCode",
                "params": [address, "latest"],
                "id": 1
            }
        )
        data = await resp.json()
        if data['result'] != '0x':
            return False
        return True
    except Exception as e:
        traceback.print_exc()
    finally:
        await client.close()
    return False


async def main():
    print(datetime.datetime.now(), 'start...')
    async for address in generate_swap_addresses():
        task = listen(address)
        asyncio.create_task(task)


if __name__ == '__main__':
    asyncio.run(main())
