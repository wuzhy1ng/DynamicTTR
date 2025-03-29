import asyncio
import json
import os.path

import websockets

MAX_NUM_ADDRESSES = 1000
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
            SWAP_ADDRESSES.add(data['address'])
            SWAP_ACTION_FILE.write(json.dumps(data))
            SWAP_ACTION_FILE.write('\n')
            SWAP_ACTION_FILE.flush()
            yield data['address']


async def main():
    async for address in generate_swap_addresses():
        task = listen(address)
        asyncio.create_task(task)


if __name__ == '__main__':
    asyncio.run(main())
