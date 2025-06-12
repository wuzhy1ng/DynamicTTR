import csv
import decimal
import os.path
from typing import Dict

import requests

from settings import HTTP_PROXY, PROJECT_PATH

# load cache if exists
_cache_price = dict()
_cache_path = os.path.join(PROJECT_PATH, 'cache')
if not os.path.exists(_cache_path):
    os.makedirs(_cache_path)
_cache_fn = os.path.join(_cache_path, 'coin_price.csv')
_is_cache_exists = os.path.exists(_cache_fn)
if _is_cache_exists:
    with open(_cache_fn, 'r', encoding='utf-8', newline='\n') as f:
        for row in csv.DictReader(f):
            key = (
                row['platform'], row['name'],
                int(int(row['timestamp']) / 100) * 100
            )
            _cache_price[key] = {
                'price': float(row['price']),
                'decimals': int(row['decimals']),
            }

# open cache file for saving cache
_cache_file = open(_cache_fn, 'a', encoding='utf-8', newline='\n')
_cache_writer = csv.writer(_cache_file)
if not _is_cache_exists:
    _cache_writer.writerow(['platform', 'name', 'timestamp', 'price', 'decimals'])
    _cache_file.flush()


def coin_price(
        platform: str,
        name: str,
        timestamp: int,
) -> Dict:
    # query the cache
    coins = '%s:%s' % (platform, name)
    _cache_price_key = (platform, name, int(timestamp / 100) * 100)
    data = _cache_price.get(_cache_price_key)
    if data is not None:
        return data

    # query the api if not cache available
    url = 'https://coins.llama.fi/prices/historical/{}/{}?searchWidth=4h'.format(
        timestamp, coins
    )
    for i in range(2):
        try:
            params = {
                "url": url,
                "headers": {'Accept': 'application/json'},
            }
            if HTTP_PROXY is not None:
                params['proxies'] = {
                    'http': HTTP_PROXY,
                    'https': HTTP_PROXY,
                }
            response = requests.get(**params)
            data = response.json()
            data = data['coins'][coins]

            # save to cache
            _cache_price[_cache_price_key] = {
                'price': data['price'],
                'decimals': data['decimals'],
            }
            _cache_writer.writerow([
                platform, name, timestamp,
                data['price'], data['decimals'],
            ])
            _cache_file.flush()

            # return the price data
            return {
                'price': data['price'],
                'decimals': data['decimals'],
            }
        except KeyError:
            # save to cache
            _cache_price[_cache_price_key] = {
                'price': -1,
                'decimals': -1,
            }
            _cache_writer.writerow([
                platform, name, timestamp,
                -1, -1,
            ])
            _cache_file.flush()
            break
        except Exception as e:
            print('error:', type(e), e, 'try again (%s / 2)' % (i + 1))
    return {
        'price': -1,
        'decimals': -1,
    }


def get_usd_value(
        contract_address: str,
        value: str,
        timestamp: int
) -> str:
    data = coin_price(
        platform='ethereum',
        name=contract_address,
        timestamp=timestamp
    )
    if data['decimals'] < 0:
        return '0'
    value = decimal.Decimal(value)
    token_decimals = decimal.Decimal(data['decimals'])
    value = value / (decimal.Decimal('10') ** token_decimals)
    price = decimal.Decimal(data['price'])
    value = value * price
    return str(value)


if __name__ == '__main__':
    print(get_usd_value(
        contract_address='0x66761fa41377003622aee3c7675fc7b5c1c2fac5',
        value='1.2363200046968E+22',
        timestamp=1639265618
    ))
