import functools
from typing import Dict

import requests


@functools.lru_cache(maxsize=2 ** 16)
def get_usd_price(contract_address: str, timestamp: int) -> Dict | None:
    response = requests.post(
        url='http://172.18.219.142:55000/api/v1/get_token_price_usd',
        json={
            "jsonrpc": "2.0",
            "id": "0",
            "method": "get_token_price_usd",
            "params": {
                "platform": "ethereum",
                "name": contract_address,
                "timestamp": timestamp,
            }
        }
    )
    data: dict = response.json()
    if not data.get('result'):
        return None
    return data['result']
