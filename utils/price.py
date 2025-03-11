import decimal
from typing import Dict

import requests


def get_usd_price(contract_address: str, timestamp: int) -> Dict | None:
    try:
        response = requests.post(
            url='http://127.0.0.1:55000/api/v1/get_token_price_usd',
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
    except Exception as _:
        return None


def get_usd_value(
        contract_address: str,
        value: str,
        timestamp: int
) -> str:
    data = get_usd_price(
        contract_address=contract_address,
        timestamp=timestamp
    )
    if data is None:
        return '0'
    value = decimal.Decimal(value)
    token_decimals = decimal.Decimal(data['decimals'])
    value = value / (decimal.Decimal('10') ** token_decimals)
    price = decimal.Decimal(data['price'])
    value = value * price
    return str(value)
