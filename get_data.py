# get the data in the first step

def fetch_last_liquidations(max_total=40000, batch=1000, market=None):
    out, skip = [], 0

    selection = """
      id hash nonce logIndex gasPrice gasUsed gasLimit blockNumber timestamp
      liquidator { id } liquidatee { id } market { id }
      positions(first: 100) {
        asset { id symbol decimals }
        side
        balance
        account { id }
        market { id }
      }
      asset { id symbol decimals }
      amount amountUSD profitUSD
    """

    # no market filter
    q_all = f"""
    query ($first: Int!, $skip: Int!) {{
      liquidates(first:$first, skip:$skip, orderBy: timestamp, orderDirection: desc) {{
        {selection}
      }}
    }}"""

    # with market filter
    q_market = f"""
    query ($first: Int!, $skip: Int!, $market: ID!) {{
      liquidates(
        first:$first, skip:$skip,
        where: {{ market: $market }},
        orderBy: timestamp, orderDirection: desc
      ) {{
        {selection}
      }}
    }}"""

    query = q_market if market else q_all

    while len(out) < max_total:
        variables = {"first": batch, "skip": skip}
        if market:
            variables["market"] = market.lower()

        resp = requests.post(ENDPOINT, json={"query": query, "variables": variables})
        resp.raise_for_status()
        payload = resp.json()

        if "errors" in payload:
            raise RuntimeError(payload["errors"])  # surface the GraphQL error

        rows = payload["data"]["liquidates"]
        out.extend(rows)
        if len(rows) < batch:
            break
        skip += batch

    return pd.json_normalize(out[:max_total], sep="_")








# get borrowers loan before liquidation using revious positions

w3 = Web3(Web3.HTTPProvider(RPC_URL))
RPS_LIMIT = 5            # ~5 calls/sec is very safe
RETRY = 3
SLEEP_BASE = 0.4         # 1/RPS_LIMIT

# ---- tiny rate limiter ----
_last = [0.0]
def rate_limit():
    now = time.time()
    delta = now - _last[0]
    if delta < SLEEP_BASE:
        time.sleep(SLEEP_BASE - delta)
    _last[0] = time.time()

def with_retry(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        for i in range(RETRY+1):
            try:
                rate_limit()
                return fn(*args, **kwargs)
            except Exception as e:
                if i == RETRY:
                    raise
                time.sleep(0.5 * (2**i) + random.random()*0.2)
    return wrapper

# ---- caches you should keep in memory during the run ----
BLOCK_HASH = {}
ASSET_LIST = {}   # key: comet_addr -> list of asset addresses

@with_retry
def get_block_hash(block_number: int) -> str:
    b = int(block_number)
    if b in BLOCK_HASH: return BLOCK_HASH[b]
    h = w3.eth.get_block(b).hash.hex()
    BLOCK_HASH[b] = h
    return h

# minimal Comet ABI as before (only calls you need)
COMET_MIN_ABI = [
  {"constant":True,"inputs":[{"name":"account","type":"address"}],"name":"borrowBalanceOf","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
  {"constant":True,"inputs":[],"name":"numAssets","outputs":[{"name":"","type":"uint8"}],"stateMutability":"view","type":"function"},
  {"constant":True,"inputs":[{"name":"i","type":"uint8"}],"name":"getAssetInfo","outputs":[
      {"name":"offset","type":"uint8"},{"name":"asset","type":"address"},{"name":"priceFeed","type":"address"},
      {"name":"scale","type":"uint64"},{"name":"borrowCollateralFactor","type":"uint64"},
      {"name":"liquidateCollateralFactor","type":"uint64"},{"name":"liquidationFactor","type":"uint64"},
      {"name":"supplyCap","type":"uint128"}],"stateMutability":"view","type":"function"},
  {"constant":True,"inputs":[{"name":"account","type":"address"},{"name":"asset","type":"address"}],
   "name":"userCollateral","outputs":[{"name":"balance","type":"uint128"},{"name":"_","type":"uint128"}],
   "stateMutability":"view","type":"function"},
  {"constant":True,"inputs":[],"name":"baseToken","outputs":[{"name":"","type":"address"}],
   "stateMutability":"view","type":"function"}
]

def comet_from_market_id(market_id: str) -> str:
    # first 40 hex chars after '0x'
    return Web3.to_checksum_address("0x" + market_id.lower().replace("0x","")[:40])

@with_retry
def get_asset_list(comet):
    if comet.address in ASSET_LIST:
        return ASSET_LIST[comet.address]
    n = comet.functions.numAssets().call()
    assets = [comet.functions.getAssetInfo(i).call()[1] for i in range(n)]
    ASSET_LIST[comet.address] = assets
    return assets

@with_retry
def call_at_block(func, block):
    return func.call(block_identifier=block)

def fetch_pre_positions_onchain(row):
    acct      = Web3.to_checksum_address(str(row["liquidatee_id"]))
    market_id = str(row["market_id"])
    comet_addr= comet_from_market_id(market_id)
    comet     = w3.eth.contract(address=comet_addr, abi=COMET_MIN_ABI)

    b_minus_1 = int(row["blockNumber"]) - 1
    _ = get_block_hash(b_minus_1)  # warms cache + validates block

    # base debt
    base_token = call_at_block(comet.functions.baseToken(), b_minus_1)
    base_debt  = call_at_block(comet.functions.borrowBalanceOf(acct), b_minus_1)

    out = [{
        "account": {"id": acct.lower()},
        "asset":   {"id": base_token.lower(), "symbol": None, "decimals": None},
        "balance": str(base_debt),
        "market":  {"id": market_id.lower()},
        "side":    "BORROWER"
    }]

    # collaterals
    assets = get_asset_list(comet)
    for a in assets:
        bal = call_at_block(comet.functions.userCollateral(acct, a), b_minus_1)[0]
        if bal == 0:  # optional: skip zeros to cut results
            continue
        out.append({
            "account": {"id": acct.lower()},
            "asset":   {"id": Web3.to_checksum_address(a).lower(), "symbol": None, "decimals": None},
            "balance": str(bal),
            "market":  {"id": market_id.lower()},
            "side":    "COLLATERAL"
        })
    return out

def add_positions_before(df, col="positions_before"):
    df = df.copy()
    results = []
    for i, row in df.iterrows():
        try:
            results.append(fetch_pre_positions_onchain(row))
        except Exception as e:
            results.append(None)
        # chunk pacing every 200 rows
        if (i+1) % 200 == 0:
            time.sleep(2)
    df[col] = results
    return df








# Get token prices

# chainlink_prices_fast.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time
import concurrent.futures as cf
import pandas as pd
from web3 import Web3
from web3.exceptions import BadFunctionCallOutput, ContractLogicError


# Put your lists here
block_numbers = [21000000, 21000010, 21000020]
token_addresses = [
    "0x514910771af9ca656af840dff83e8264ecf986ca",  # LINK
    "0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0",  # wstETH
    "0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf",
    "0xc00e94cb662c3520282e6f5717214004a7f26888",  # COMP
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",  # WBTC (NOT COMP)
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
    "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",  # UNI
    "0x18084fba666a33d37592fa2633fd49a74dd93a88",  # tBTC
]

# Concurrency & pacing (be nice to Alchemy)
MAX_WORKERS = 8
CALL_SLEEP_SECONDS = 0.01
MAX_RETRIES = 5

# ========= ADDRESSES / ABIs =========
# Chainlink Feed Registry (mainnet)
FEED_REGISTRY = Web3.to_checksum_address("0x47Fb2585D2C56Fe188D0E6ec628a38b74fCeeeDf")
ADDR_USD = Web3.to_checksum_address("0x0000000000000000000000000000000000000348")
ADDR_ETH = Web3.to_checksum_address("0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE")

# Common tokens
WETH  = Web3.to_checksum_address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")
WBTC  = Web3.to_checksum_address("0x2260fac5e5542a773aa44fbcfedf7c193bc2c599")
TBTC  = Web3.to_checksum_address("0x18084fba666a33d37592fa2633fd49a74dd93a88")
WSTETH= Web3.to_checksum_address("0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0")
STETH = Web3.to_checksum_address("0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84")  # Lido stETH

# Chainlink Aggregator proxies (stable, historical-friendly)
# ETH/USD (mainnet)
AGG_ETH_USD = Web3.to_checksum_address("0x5f4ec3df9cbd43714fe2740f5e3616155c5b8419")
# BTC/USD (mainnet)
AGG_BTC_USD = Web3.to_checksum_address("0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88c")

# Minimal ABIs
FEED_REGISTRY_ABI = [
    {
        "inputs": [{"internalType":"address","name":"base","type":"address"},
                   {"internalType":"address","name":"quote","type":"address"}],
        "name":"latestRoundData",
        "outputs":[
            {"internalType":"uint80","name":"roundId","type":"uint80"},
            {"internalType":"int256","name":"answer","type":"int256"},
            {"internalType":"uint256","name":"startedAt","type":"uint256"},
            {"internalType":"uint256","name":"updatedAt","type":"uint256"},
            {"internalType":"uint80","name":"answeredInRound","type":"uint80"},
        ],
        "stateMutability":"view","type":"function"
    },
    {
        "inputs": [{"internalType":"address","name":"base","type":"address"},
                   {"internalType":"address","name":"quote","type":"address"}],
        "name":"decimals","outputs":[{"internalType":"uint8","name":"","type":"uint8"}],
        "stateMutability":"view","type":"function"
    },
]

AGGREGATOR_V3_ABI = [
    {
        "inputs": [],
        "name": "latestRoundData",
        "outputs": [
            {"internalType":"uint80","name":"roundId","type":"uint80"},
            {"internalType":"int256","name":"answer","type":"int256"},
            {"internalType":"uint256","name":"startedAt","type":"uint256"},
            {"internalType":"uint256","name":"updatedAt","type":"uint256"},
            {"internalType":"uint80","name":"answeredInRound","type":"uint80"},
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "decimals",
        "outputs": [{"internalType":"uint8","name":"","type":"uint8"}],
        "stateMutability": "view",
        "type": "function"
    }
]

# wstETH ratio
WSTETH_ABI = [
    {
        "inputs": [],
        "name": "stEthPerToken",
        "outputs": [{"internalType":"uint256","name":"","type":"uint256"}],
        "stateMutability": "view",
        "type": "function"
    }
]

@dataclass
class RoundData:
    answer: Optional[float]
    updated_at: Optional[int]

class PriceFetcher:
    def __init__(self, w3: Web3):
        self.w3 = w3
        self.registry = w3.eth.contract(address=FEED_REGISTRY, abi=FEED_REGISTRY_ABI)
        self.agg_eth_usd = w3.eth.contract(address=AGG_ETH_USD, abi=AGGREGATOR_V3_ABI)
        self.agg_btc_usd = w3.eth.contract(address=AGG_BTC_USD, abi=AGGREGATOR_V3_ABI)
        self.wsteth = w3.eth.contract(address=WSTETH, abi=WSTETH_ABI)
        # caches
        self._dec_cache: Dict[Tuple[str, str], Optional[int]] = {}
        self._reg_price_cache: Dict[Tuple[int, str, str], Optional[RoundData]] = {}
        self._agg_price_cache: Dict[Tuple[int, str], Optional[RoundData]] = {}
        self._ethusd_cache: Dict[int, Optional[RoundData]] = {}
        self._btcusd_cache: Dict[int, Optional[RoundData]] = {}
        self._wst_ratio_cache: Dict[int, Optional[float]] = {}

    def _sleep(self):  # small, friendly pacing
        time.sleep(CALL_SLEEP_SECONDS)

    def _retry(self, func, *args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                self._sleep()
                return func(*args, **kwargs)
            except Exception:
                time.sleep(0.2 * (attempt + 1))
        raise

    # ---- Aggregator helpers (fast + historical) ----
    def _agg_decimals(self, agg):
        return self._retry(lambda: agg.functions.decimals().call())

    def _agg_latest(self, agg, block):
        data = self._retry(lambda: agg.functions.latestRoundData().call(block_identifier=block))
        if not data: return None
        _, ans, _, upd, _ = data
        dec = self._retry(lambda: agg.functions.decimals().call())
        if ans and int(ans) > 0:
            return RoundData(float(ans) / 10**dec, upd)
        return None

    def _eth_usd(self, block) -> Optional[RoundData]:
        if block not in self._ethusd_cache:
            try:
                self._ethusd_cache[block] = self._agg_latest(self.agg_eth_usd, block)
            except Exception:
                self._ethusd_cache[block] = None
        return self._ethusd_cache[block]

    def _btc_usd(self, block) -> Optional[RoundData]:
        if block not in self._btcusd_cache:
            try:
                self._btcusd_cache[block] = self._agg_latest(self.agg_btc_usd, block)
            except Exception:
                self._btcusd_cache[block] = None
        return self._btcusd_cache[block]

    # ---- Feed Registry helpers (for non-pegs) ----
    def _reg_decimals(self, base, quote) -> Optional[int]:
        key = (base, quote)
        if key not in self._dec_cache:
            try:
                self._dec_cache[key] = self.registry.functions.decimals(base, quote).call()
            except Exception:
                self._dec_cache[key] = None
        return self._dec_cache[key]

    def _reg_price(self, base, quote, block) -> Optional[RoundData]:
        key = (block, base, quote)
        if key in self._reg_price_cache:
            return self._reg_price_cache[key]
        try:
            data = self.registry.functions.latestRoundData(base, quote).call(block_identifier=block)
            _, ans, _, upd, _ = data
            dec = self._reg_decimals(base, quote)
            rd = None
            if ans and dec is not None and int(ans) > 0:
                rd = RoundData(float(ans) / 10**dec, upd)
        except (ContractLogicError, BadFunctionCallOutput):
            rd = None
        except Exception:
            rd = None
        self._reg_price_cache[key] = rd
        return rd

    # ---- wstETH ratio ----
    def _wsteth_ratio(self, block) -> Optional[float]:
        if block in self._wst_ratio_cache:
            return self._wst_ratio_cache[block]
        try:
            raw = self.wsteth.functions.stEthPerToken().call(block_identifier=block)
            self._wst_ratio_cache[block] = float(raw) / 1e18
        except Exception:
            self._wst_ratio_cache[block] = None
        return self._wst_ratio_cache[block]

    # ---- public: USD price at a block ----
    def price_usd_at(self, token_addr: str, block: int) -> Optional[float]:
        token = Web3.to_checksum_address(token_addr)

        # Pegged wrappers
        if token == WETH:
            ethusd = self._eth_usd(block)
            return ethusd.answer if ethusd else None

        if token in (WBTC, TBTC):
            btcusd = self._btc_usd(block)
            return btcusd.answer if btcusd else None

        if token == WSTETH:
            # Try wstETH/ETH * ETH/USD via registry, else use stEthPerToken * ETH/USD
            t_eth = self._reg_price(token, ADDR_ETH, block)
            ethusd = self._eth_usd(block)
            if t_eth and ethusd and t_eth.answer and ethusd.answer:
                return t_eth.answer * ethusd.answer

            ratio = self._wsteth_ratio(block)  # stETH per 1 wstETH
            if ratio and ethusd and ethusd.answer:
                # If stETH/ETH feed not reliably available at this block, assume stETH≈ETH peg (long-run ~1)
                # This keeps it Chainlink-based for ETH/USD and uses on-chain ratio to scale.
                return ratio * ethusd.answer
            return None

        # For general ERC-20s:
        # 1) token/USD via registry at block
        d = self._reg_price(token, ADDR_USD, block)
        if d and d.answer:
            return d.answer
        # 2) token/ETH via registry * ETH/USD
        t_eth = self._reg_price(token, ADDR_ETH, block)
        ethusd = self._eth_usd(block)
        if t_eth and ethusd and t_eth.answer and ethusd.answer:
            return t_eth.answer * ethusd.answer
        return None


def fetch_prices_dataframe(rpc_url: str, tokens: List[str], blocks: List[int]) -> pd.DataFrame:
    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 30}))
    if not w3.is_connected():
        raise RuntimeError("Web3 not connected. Check RPC_URL or network.")

    f = PriceFetcher(w3)

    def row_for_block(b: int) -> Dict[str, Optional[float]]:
        blk_prev = int(b) - 1
        out = {"block": b}
        with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(f.price_usd_at, addr, blk_prev): Web3.to_checksum_address(addr) for addr in tokens}
            for fut, col in futs.items():
                try:
                    out[col] = fut.result(timeout=20)
                except Exception:
                    out[col] = None
        return out

    rows = [row_for_block(b) for b in blocks]
    df = pd.DataFrame(rows)
    ordered = ["block"] + [Web3.to_checksum_address(a) for a in tokens]
    return df.reindex(columns=ordered)






# get the parameters: liquidity factor and store_front_price_factor 

from typing import List, Dict, Union
import numpy as np
import pandas as pd
from web3 import Web3

DEFAULT_ASSETS: Dict[str, str] = {
    "LINK":  "0x514910771AF9Ca656af840dff83E8264EcF986CA",
    "wstETH":"0x7f39c581f595b53c5cb5bb7d1409e5c3f6b7c53f",
    "cbBTC": "0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf",
    "COMP":  "0xc00e94cb662c3520282e6f5717214004a7f26888",
    "WBTC":  "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
    "WETH":  "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "UNI":   "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
    "tBTC":  "0x18084fba666a33d37592fa2633fd49a74dd93a88",
}

COMET_ABI = [
    {"name": "storeFrontPriceFactor", "inputs": [], "outputs": [{"type": "uint256"}],
     "stateMutability": "view", "type": "function"},
    {"name": "numAssets", "inputs": [], "outputs": [{"type": "uint256"}],
     "stateMutability": "view", "type": "function"},
    {"name": "getAssetInfo", "inputs": [{"name": "i", "type": "uint256"}],
     "outputs": [{"components": [
        {"name": "asset", "type": "address"},
        {"name": "priceFeed", "type": "address"},
        {"name": "scale", "type": "uint256"},
        {"name": "borrowCollateralFactor", "type": "uint256"},
        {"name": "liquidationFactor", "type": "uint256"},
        {"name": "supplyCap", "type": "uint256"}],
        "type": "tuple"}],
     "stateMutability": "view", "type": "function"},
    {"name": "getAssetInfoByAddress", "inputs": [{"name": "asset", "type": "address"}],
     "outputs": [{"components": [
        {"name": "asset", "type": "address"},
        {"name": "priceFeed", "type": "address"},
        {"name": "scale", "type": "uint256"},
        {"name": "borrowCollateralFactor", "type": "uint256"},
        {"name": "liquidationFactor", "type": "uint256"},
        {"name": "supplyCap", "type": "uint256"}],
        "type": "tuple"}],
     "stateMutability": "view", "type": "function"},
]

WAD = 1e18

def _normalize_block_id(b: Union[int, str, np.integer]) -> Union[int, str]:
    if isinstance(b, str):
        s = b.lower()
        if s == "latest":
            return "latest"
        return int(b, 0)
    if isinstance(b, (np.integer,)):
        return int(b)
    return int(b)

def get_comet_params_by_blocks(
    rpc_url: str,
    comet_proxy: str,
    blocks: List[Union[int, str]],
    assets: Dict[str, str] = None,
    debug: bool = True,
) -> pd.DataFrame:
    if assets is None:
        assets = DEFAULT_ASSETS

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    comet_addr = Web3.to_checksum_address(comet_proxy)
    comet = w3.eth.contract(address=comet_addr, abi=COMET_ABI)
    rows = []

    for blk_raw in blocks:
        blk = _normalize_block_id(blk_raw)

        # pre-deploy check
        code = w3.eth.get_code(comet_addr, block_identifier=blk)
        if not code or code == b"\x00":
            if debug: print(f"[blk {blk}] no code at this block (pre-deploy)")
            continue

        # storefront
        try:
            sfpf = comet.functions.storeFrontPriceFactor().call(block_identifier=blk) / WAD
        except Exception as e:
            if debug: print(f"[blk {blk}] storeFrontPriceFactor error: {e}")
            sfpf = None

        # listed assets
        listed = set()
        try:
            n = int(comet.functions.numAssets().call(block_identifier=blk))
            for i in range(n):
                info = comet.functions.getAssetInfo(i).call(block_identifier=blk)
                listed.add(Web3.to_checksum_address(info[0]))
        except Exception as e:
            if debug: print(f"[blk {blk}] numAssets/getAssetInfo error: {e}")

        block_rows = []
        for sym, addr in assets.items():
            a = Web3.to_checksum_address(addr)
            try:
                info = comet.functions.getAssetInfoByAddress(a).call(block_identifier=blk)
                liq = float(info[4]) / WAD
                disc = (sfpf * (1 - liq)) if sfpf is not None else None
                note = None
            except Exception as e:
                if debug: print(f"[blk {blk}] {sym} getAssetInfoByAddress error: {e}")
                liq, disc = None, None
                note = "not listed at block" if a not in listed else f"reverted: {e}"

            block_rows.append({
                "block": blk, "asset": sym,
                "listed_at_block": a in listed,
                "storeFrontPriceFactor": sfpf,
                "liquidationFactor": liq,
                "discountFactor": disc,
                "note": note,
            })

        # 🔹 fill wstETH with WETH values if missing
        df_blk = pd.DataFrame(block_rows)
        weth_liq = df_blk.loc[df_blk["asset"] == "WETH", "liquidationFactor"].values
        if len(weth_liq) and not np.isnan(weth_liq[0]):
            df_blk.loc[df_blk["asset"] == "wstETH", "liquidationFactor"] = weth_liq[0]
            df_blk.loc[df_blk["asset"] == "wstETH", "discountFactor"] = (
                df_blk.loc[df_blk["asset"] == "wstETH", "storeFrontPriceFactor"].values[0]
                * (1 - weth_liq[0])
            )
            df_blk.loc[df_blk["asset"] == "wstETH", "note"] = "set equal to WETH"

        rows.append(df_blk)

    return pd.concat(rows, ignore_index=True)




































