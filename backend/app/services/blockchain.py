import hashlib
import time
import json
import uuid
import os
import logging
from web3 import Web3
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Load Blockchain Configuration
RPC_URL = os.getenv("BLOCKCHAIN_RPC_URL", "")
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS", "")
PRIVATE_KEY = os.getenv("WALLET_PRIVATE_KEY", "")

# The ABI defines how Python talks to the Smart Contract
CONTRACT_ABI = [
    {
        "inputs": [
            {"internalType": "string", "name": "_reportId", "type": "string"},
            {"internalType": "string", "name": "_documentHash", "type": "string"},
            {"internalType": "string", "name": "_reportHash", "type": "string"}
        ],
        "name": "logReport",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "string", "name": "_reportId", "type": "string"},
            {"internalType": "string", "name": "_reportHash", "type": "string"}
        ],
        "name": "verifyReport",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function"
    }
]

def generate_report_hash(report_data: dict) -> str:
    """Generates a deterministic SHA-256 hash of the verification report."""
    report_string = json.dumps(report_data, sort_keys=True)
    return hashlib.sha256(report_string.encode('utf-8')).hexdigest()

def generate_document_hash(image_bytes: bytes) -> str:
    """Generates a SHA-256 hash of the uploaded document image."""
    return hashlib.sha256(image_bytes).hexdigest()

def log_to_blockchain(report_data: dict, image_bytes: bytes) -> dict:
    """
    Logs the hash to an Ethereum/Polygon Smart Contract.
    Falls back to simulation if Web3 credentials are not set or encounter error.
    """
    report_id = f"RX-{str(uuid.uuid4())[:8].upper()}"
    doc_hash = generate_document_hash(image_bytes)
    rep_hash = generate_report_hash(report_data)
    
    # Check if Web3 credentials exist
    if RPC_URL and CONTRACT_ADDRESS and PRIVATE_KEY and "YAHAN_" not in PRIVATE_KEY:
        try:
            w3 = Web3(Web3.HTTPProvider(RPC_URL))
            if w3.is_connected():
                account = w3.eth.account.from_key(PRIVATE_KEY)
                contract = w3.eth.contract(address=w3.to_checksum_address(CONTRACT_ADDRESS), abi=CONTRACT_ABI)
                
                # Fetch dynamic gas fees
                latest_block = w3.eth.get_block('latest')
                base_fee = latest_block.get('baseFeePerGas', w3.to_wei('1', 'gwei'))
                priority_fee = w3.to_wei('2', 'gwei')
                max_fee = base_fee * 2 + priority_fee
                
                # Dynamically estimate gas with buffer
                try:
                    estimated = contract.functions.logReport(report_id, doc_hash, rep_hash).estimate_gas({'from': account.address})
                    gas_limit = int(estimated * 1.3)
                except Exception:
                    gas_limit = 450000
                
                nonce = w3.eth.get_transaction_count(account.address)
                tx = contract.functions.logReport(report_id, doc_hash, rep_hash).build_transaction({
                    'chainId': w3.eth.chain_id,
                    'gas': gas_limit,
                    'maxFeePerGas': max_fee,
                    'maxPriorityFeePerGas': priority_fee,
                    'nonce': nonce,
                })
                
                signed_tx = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
                raw_tx = getattr(signed_tx, 'raw_transaction', getattr(signed_tx, 'rawTransaction', None))
                tx_hash = w3.eth.send_raw_transaction(raw_tx)
                tx_hash_hex = tx_hash.hex()
                if not tx_hash_hex.startswith("0x"):
                    tx_hash_hex = "0x" + tx_hash_hex
                
                logger.info(f"Broadcasted to Ethereum Sepolia: {tx_hash_hex}")
                
                return {
                    "report_id": report_id,
                    "document_hash": doc_hash,
                    "report_hash": rep_hash,
                    "blockchain_status": "VERIFIED (SEPOLIA TESTNET)",
                    "transaction_id": tx_hash_hex,
                    "timestamp": time.strftime("%d %b %Y %H:%M:%S")
                }
        except Exception as e:
            logger.error(f"Blockchain Web3 Error: {e}", exc_info=True)
            # Fallback to simulation if transaction fails
            pass
            
    # --- SIMULATION FALLBACK ---
    time.sleep(1)
    simulated_tx_id = "0x" + hashlib.sha256(str(time.time()).encode()).hexdigest()
    
    return {
        "report_id": report_id,
        "document_hash": doc_hash,
        "report_hash": rep_hash,
        "blockchain_status": "VERIFIED (SIMULATED)",
        "transaction_id": simulated_tx_id,
        "timestamp": time.strftime("%d %b %Y %H:%M:%S")
    }
