import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from web3 import Web3
from backend.app import w3, registry_contract, ORACLE_PRIVATE_KEY

site_id = "TEST-B0-004"
user_address = "0xb7f611111AC0228799bFBbF5BEbf1E6B6ddD4e83"

print("1. Registering project on-chain directly...")
oracle_account = w3.eth.account.from_key(ORACLE_PRIVATE_KEY)

# registerProject(string siteId, address owner, string latitude, string longitude, uint256 areaHectares)
tx = registry_contract.functions.registerProject(
    site_id,
    "21.9",
    "88.5",
    500
).build_transaction({
    'from': oracle_account.address,
    'nonce': w3.eth.get_transaction_count(oracle_account.address),
    'gasPrice': w3.eth.gas_price
})

signed_tx = w3.eth.account.sign_transaction(tx, private_key=ORACLE_PRIVATE_KEY)
tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
print(f"Registration tx sent: {tx_hash.hex()}")
w3.eth.wait_for_transaction_receipt(tx_hash)
print("Registered!")

print("2. Issuing credits directly...")
# verifyAndIssueCredits(string siteId, uint256 carbonTons, string ipfsProofHash)
tx2 = registry_contract.functions.verifyAndIssueCredits(
    site_id,
    15000,
    "QmTestHash..."
).build_transaction({
    'from': oracle_account.address,
    'nonce': w3.eth.get_transaction_count(oracle_account.address),
    'gasPrice': w3.eth.gas_price
})
signed_tx2 = w3.eth.account.sign_transaction(tx2, private_key=ORACLE_PRIVATE_KEY)
tx_hash2 = w3.eth.send_raw_transaction(signed_tx2.raw_transaction)
w3.eth.wait_for_transaction_receipt(tx_hash2)
print("Issued credits!")

print("3. Querying /verify endpoint locally...")
from fastapi.testclient import TestClient
from backend.app import app
client = TestClient(app)
resp = client.get(f"/verify/{site_id}")
print(f"Status: {resp.status_code}")
if "Project Not Found" in resp.text:
    print("WARNING: Rendered error page!")
else:
    print("SUCCESS: Rendered valid template!")
