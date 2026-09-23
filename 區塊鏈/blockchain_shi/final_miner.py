import rsa
import hashlib
import time
import socket
import threading
import sys
import pickle
import os
import pandas as pd

# Step1 *----------------------------------------------------*

class Transaction:
    def __init__(self, sender, receiver, amounts, fee, message):
        self.sender = sender #sender address
        self.receiver = receiver #receiver address
        self.amounts = amounts   #crytocurrency amount
        self.fee = fee #transaction fee for miner 
        self.message = message
        
class Block:
    def __init__(self, previous_hash, difficulty, miner, miner_rewards):
        self.previous_hash = previous_hash  #前一區塊的hash value 
        self.hash = ''  #這個區塊的hash value
        self.difficulty = difficulty   #difficulity target 
        self.nonce = 0  #解開區塊的鑰匙
        self.timestamp = int(time.time())  #出塊的時間
        self.transactions = [] #這個區塊內存放的所有交易
        self.miner = miner  #挖出此區塊的miner(address)
        self.miner_rewards = miner_rewards #miner reward

# Step2 *----------------------------------------------------*

class BlockChain:
    def __init__(self):
        self.adjust_difficulty_blocks = 10  #每挖出幾個區塊後調整難度
        self.difficulty = 5  #挖出區塊的難度
        self.block_time = 30  #預測平均出塊時間
        self.mining_rewards = 10  #出塊獎勵
        self.block_limitation = 32  #每個區塊可容納得交易數量上限
        self.chain = [] #區塊鏈-存放所有區塊鏈
        self.pending_transactions = [] #等待打包近區塊的交易池
        # For P2P connection
        self.socket_host = "127.0.0.1" # connect to this computer
        self.socket_port = 2024
        self.start_socket_server()
        # maintain the p2p network
        self.address_table = [] # [addr, apply tokens] (存放網路中所有用戶的地址)
        # temp
        self.mining_flag = 1  #mining control
        # file name for the final output !
        self.ChainFileName='blockchain.csv'
    # Step3 *----------------------------------------------------*
    # 產生區塊的雜湊值
    
    def get_transactions_string(self, block):
        transaction_str = ''
        for transaction in block.transactions:
            transaction_str += self.transaction_to_string(transaction)
        return transaction_str

    def get_hash(self, block, nonce):
        s = hashlib.sha1()
        s.update(
            (
                block.previous_hash
                + str(block.timestamp)
                + self.get_transactions_string(block)
                + str(nonce)
            ).encode("utf-8")
        )
        h = s.hexdigest()
        return h
    
    def create_genesis_block(self):
        print("Create genesis block...")
        new_block = Block('Hello World!', self.difficulty, 'lkm543', self.mining_rewards)
        new_block.hash = self.get_hash(new_block, 0)
        self.chain.append(new_block)     

    # Step4 *----------------------------------------------------*
    # 挖掘區塊
    
    def add_transaction_to_block(self, block):
        # Get the transaction with highest fee by block_limitation
        self.pending_transactions.sort(key=lambda x: x.fee, reverse=True)
        if len(self.pending_transactions) > self.block_limitation:
            transcation_accepted = self.pending_transactions[:self.block_limitation]
            self.pending_transactions = self.pending_transactions[self.block_limitation:]
        else:
            transcation_accepted = self.pending_transactions
            self.pending_transactions = []
        block.transactions = transcation_accepted
        
    def mine_block(self, miner):
        start = time.process_time()

        last_block = self.chain[-1]
        new_block = Block(last_block.hash, self.difficulty, miner, self.mining_rewards)

        self.add_transaction_to_block(new_block)
        new_block.previous_hash = last_block.hash
        new_block.difficulty = self.difficulty
        new_block.hash = self.get_hash(new_block, new_block.nonce)

        while new_block.hash[0: self.difficulty] != '0' * self.difficulty:
            new_block.nonce += 1
            new_block.hash = self.get_hash(new_block, new_block.nonce)

        time_consumed = round(time.process_time() - start, 5)
        print(f"Hash found: {new_block.hash} @ difficulty {self.difficulty}, time cost: {time_consumed}s")
        self.chain.append(new_block)
        
    def adjust_difficulty(self):
        if len(self.chain) % self.adjust_difficulty_blocks != 0:
            return self.difficulty
        elif len(self.chain) <= self.adjust_difficulty_blocks:
            return self.difficulty
        else:
            start = self.chain[-1*self.adjust_difficulty_blocks-1].timestamp
            finish = self.chain[-1].timestamp
            average_time_consumed = round((finish - start) / (self.adjust_difficulty_blocks), 2)
            if average_time_consumed > self.block_time:
                print(f"Average block time:{average_time_consumed}s. Lower the difficulty")
                self.difficulty -= 1
            else:
                print(f"Average block time:{average_time_consumed}s. High up the difficulty")
                self.difficulty += 1
    
    # Step5 *----------------------------------------------------*   
    
    def start_socket_server(self):
        t = threading.Thread(target=self.wait_for_socket_connection)
        t.start()

    def wait_for_socket_connection(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.socket_host, self.socket_port))
            s.listen()
            while True:
                conn, address = s.accept()

                client_handler = threading.Thread(
                    target=self.receive_socket_message,
                    args=(conn, address)
                )
                client_handler.start()
                
    # Step6 *----------------------------------------------------*
                
    def receive_socket_message(self, connection, address):
        with connection:
            print(f'Connected by: {address}')
            while True:
                try:
                    message = connection.recv(1024)
                    parsed_message = pickle.loads(message) # decode message -> dict{'request':..., 'date':...}
                    print("parsed_message : ", parsed_message)
                except Exception:
                    #print("Something wrong while decode the message !!!")
                    #print(parsed_message)
                    #os._exit(0)
                    pass
                
                if message:
                    if parsed_message["request"] == "get_balance":
                        print("Start to get the balance for client...")
                        address = parsed_message["address"]
                        balance = self.get_balance(address)
                        response = {
                            "reply": "get_balance",
                            "data": balance
                        }
                        connection.send(pickle.dumps(response))

                    elif parsed_message["request"] == "add_address":
                        print("Put client's address into database")
                        temp_list=[]
                        for i in self.address_table:
                            temp_list.append(i[0])    
                        self.address_table.append([parsed_message['data'], 0]) # store the address into address_table 
                        response = {
                            "reply":"update_address",
                            "data":temp_list
                        }
                        connection.send(pickle.dumps(response))

                    elif parsed_message["request"] == "transaction":
                        print("Start to transaction for client...")
                        print("The message: ")
                        print(parsed_message)
                        address = parsed_message["address"]
                        new_transaction = self.initialize_transaction(
                            parsed_message["address"], 
                            parsed_message["receiver"],
                            int(parsed_message["amount"]),
                            int(parsed_message["fee"]),
                            parsed_message["comment"]
                        )
                        
                        if type(new_transaction) == type(""):
                            result_message = "The Transaction is rejected by miner, since y balance is not enough."
                        else:
                            result_message = self.add_transaction(
                                new_transaction,
                                parsed_message["signature"]
                            )
                        
                        response = {
                            "reply": "transaction",
                            "data": result_message
                        }
                        connection.send(pickle.dumps(response))

                    elif parsed_message["request"] == "apply":
                        # data : addr, amount
                        temp_list = parsed_message["data"]
                        for i in self.address_table:
                            if temp_list[0] == i[0]:
                                i[1] += temp_list[1]
                                break
                        for i in self.address_table:
                            print(i)

                    # 接收到同步區塊的請求
                    elif parsed_message["request"] == "clone_blockchain":
                        print(f"[*] Receive blockchain clone request by {address}...")
                        message = {
                            "request": "upload_blockchain",
                            "blockchain_data": self
                        }
                        connection.sendall(pickle.dumps(message))
                        continue
                        
                    elif parsed_message['request'] == "close":
                        print(f"address '{address}' has disconnected to the miner.")
                        time.sleep(3)
                        connection.close()
                        break
                    
                    else:
                        response = {
                            "message": "Unknown command."
                        }
    
    # 查詢指定地址的帳戶餘額                    
    def get_balance(self, account):
        balance = 0
        for block in self.chain:
            # Check miner reward
            miner = False
            if block.miner == account:
                miner = True
                balance += block.miner_rewards
            for transaction in block.transactions:
                if miner:
                    balance += transaction.fee
                if transaction.sender == account:
                    balance -= transaction.amounts
                    balance -= transaction.fee
                elif transaction.receiver == account:
                    balance += transaction.amounts
    
        # *-----------ToDo-----------*
        # Traverse the blockchain to get the balance 
        ## for block in self.chain:
        
        # Add the test token that the customer had already applied
        for i in self.address_table:
            if account == i[0]:
                balance += i[1]
                break

        return balance
    
    def initialize_transaction(self, sender, receiver, amount, fee, message):

        if self.get_balance(sender) < amount + fee:
            print("Balance not enough!")
            return False 
        new_transaction = Transaction(sender, receiver, amount, fee, message)
        return new_transaction
        # *-----------ToDo-----------*
        # 使用 get_balance 確認交易發起方有足夠的帳戶餘額完成這筆交易!
        ## 若帳戶餘額不足請回傳 "Balance not enough!"
        ## 若帳戶餘額足夠則回傳一個 Transaction 物件
        pass # delete the instr. after you finishing the code.
    
    def transaction_to_string(self, transaction):
        transaction_dict = {
            'sender': str(transaction.sender),
            'receiver': str(transaction.receiver),
            'amounts': transaction.amounts,
            'fee': transaction.fee,
            'message': transaction.message
        }
        return str(transaction_dict)

    def add_transaction(self, transaction, signature):
        public_key = '-----BEGIN RSA PUBLIC KEY-----\n'
        public_key += transaction.sender
        public_key += '\n-----END RSA PUBLIC KEY-----\n'
        public_key_pkcs = rsa.PublicKey.load_pkcs1(public_key.encode('utf-8'))
        transaction_str = self.transaction_to_string(transaction)
        if transaction.fee + transaction.amounts > self.get_balance(transaction.sender):
            print("Balance not enough!")
            return "Balance not enough!"
        try:
            # 驗證發送者
            rsa.verify(transaction_str.encode('utf-8'), signature, public_key_pkcs)
            print("Transaction was verified! It may be packed into block later.")
            self.pending_transactions.append(transaction)
            return True
        except Exception:
            print("RSA Verified wrong!")
            return "RSA Verified wrong!"
        
    # Step7 *----------------------------------------------------*
        
    def generate_address(self):
        public, private = rsa.newkeys(512)
        public_key = public.save_pkcs1()
        private_key = private.save_pkcs1()
        
        return self.get_address_from_public(public_key), private_key
    
    def get_address_from_public(self, public):
        address = str(public).replace('\\n','')
        address = address.replace("b'-----BEGIN RSA PUBLIC KEY-----", '')
        address = address.replace("-----END RSA PUBLIC KEY-----'", '')
        address = address.replace(' ', '')
        print('Address:', address)
        self.address_table.append([address, 0]) # 將 miner 的 address 存入 table
        return address
    
    
    def start(self):
        address, private = self.generate_address()
        self.create_genesis_block()
        if True:
            t = threading.Thread(target=self.interrup_control)
            t.start()

            while(self.mining_flag == 1):            
                self.mine_block(address)
                print("balance amount: ",self.get_balance(address))
                self.adjust_difficulty()
       
    def interrup_control(self):
        self.mining_flag = int(input("In interrup_control function : input value 0 to interrup current work on mining new block. "))
 
                      

if __name__ == '__main__':
    block = BlockChain()
    block.start()
    
    print("input value 0 to stop the entire application.")
    control = int(input())
    if control==0:
        temp_list = []
        for save_block in block.chain:
            temp_list.append([save_block.previous_hash, save_block.difficulty, save_block.transactions,
                                save_block.hash, save_block.timestamp])
        df = pd.DataFrame(temp_list)
        df.to_csv( block.ChainFileName  , index= False ,header=None)
        os._exit(0)
        