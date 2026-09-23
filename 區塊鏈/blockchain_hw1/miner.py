import time
import hashlib
import socket
import threading
import pickle
import rsa
import os
import pandas as pd

# Step 1: Define the class about the transaction and the block
class Transaction:
    def __init__(self, sender, receiver, amounts, fee, message):
        self.sender = sender      # sender address
        self.receiver = receiver  # receiver address
        self.amounts = amounts    # cryptocurrency amount
        self.fee = fee            # transaction fee for the miner
        self.message = message

class Block:
    def __init__(self, previous_hash, difficulty, miner, miner_rewards):
        self.previous_hash = previous_hash  # 前一個區塊的 hash value
        self.hash = ''                      # 這個區塊的 hash value
        self.difficulty = difficulty        # 這個區塊的挖掘難度
        self.nonce = 0                      # 解開區塊的鑰匙
        self.timestamp = int(time.time())   # 出塊的時間戳記
        self.transactions = []              # 這個區塊所存放的所有交易
        self.miner = miner                  # 挖出此區塘的碳工(地址)
        self.miner_rewards = miner_rewards  # 碳工挖出區塊所能獲得的獎勵

# Step 2: Define the class about the blockchain
class BlockChain:
    def __init__(self):
        ## block info
        self.adjust_difficulty_blocks = 10    # 每挖出幾個區塊後調整難度
        self.difficulty = 5                   # 挖出區塊的難度
        self.block_time = 30                  # 預設的平均出塊時間
        self.mining_rewards = 10              # 出塊獎勵
        self.block_limitation = 32            # 每個區塊可以容納的交易數量上限
        self.chain = []                       # 區塊鏈-存放所有區塊鍵
        self.pending_transactions = []        # 等待打包進區塊的交易池
        ## For P2P connection
        self.socket_host = "127.0.0.1"        # 本機電腦IP
        self.socket_port = 2024               # 本機電腦Port
        self.address_table = []               # 存放網絡中所有用戶的地址
        self.start_socket_server()            # 啟動socket
        self.mining_flag = 1                  # mining control
        self.ChainFileName = 'blockchain.csv'  # file name for the final output

    # Step3: Create the “genesis block”
    def transaction_to_string(self, transaction):
        transaction_dict = {
            'sender': str(transaction.sender),
            'receiver': str(transaction.receiver),
            'amounts': transaction.amounts,
            'fee': transaction.fee,
            'message': transaction.message
        }
        return str(transaction_dict)

    def get_transactions_string(self, block):
        transaction_str = ''
        for transaction in block.transactions:
            transaction_str += self.transaction_to_string(transaction)
        return transaction_str
    
    def get_hash(self, block, nonce):
        s = hashlib.sha1()
        s.update(
            (
                block.previous_hash + str(block.timestamp) + self.get_transactions_string(block) + str(nonce)
            ).encode("utf-8")
        )
        h = s.hexdigest() # 將區塊的hashvalue以16進制表示
        return h

    def create_genesis_block(self):
        print("Create genesis block...")
        new_block = Block('Hello World!', self.difficulty, 'First miner', self.mining_rewards)
        new_block.hash = self.get_hash(new_block, 0)
        self.chain.append(new_block)

    # Step4: Mine the block and also dynamically adjust the difficulty
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
        start = time.process_time() # 開始計時
        last_block = self.chain[-1] # 取得最後一個區塊
        new_block = Block(last_block.hash, self.difficulty, miner, self.mining_rewards) # 創建新區塊
        self.add_transaction_to_block(new_block) # 將交易添加到新區塊

        # 設定新區塊的前一個區塊hash value和難度
        new_block.previous_hash = last_block.hash
        new_block.difficulty = self.difficulty
        new_block.hash = self.get_hash(new_block, new_block.nonce) # 計算新區塊的hash value

        # 挖礦，直到找到符合難度目標的hash value
        while new_block.hash[0: self.difficulty] != '0' * self.difficulty:
            new_block.nonce += 1
            new_block.hash = self.get_hash(new_block, new_block.nonce)

        time_consumed = round(time.process_time() - start, 5) # 計算挖礦所花費的時間
        print(f"Hash found: {new_block.hash} @ difficulty {self.difficulty}, time cost: {time_consumed}s") # 輸出挖到的區塊hash value、難度和挖礦所花費的時間
        self.chain.append(new_block) # 將新區塊添加到區塊鏈中

    def adjust_difficulty(self):
        # 如果不是調整難度的時候，保持當前難度不變
        if len(self.chain) % self.adjust_difficulty_blocks != 0:
            return self.difficulty
        
        # 如果區塊鏈的長度小於等於調整難度的區塊數，保持當前難度不變
        elif len(self.chain) <= self.adjust_difficulty_blocks:
            return self.difficulty
        
        else:
            # 計算調整難度所需的時間區間
            start = self.chain[-1 * self.adjust_difficulty_blocks - 1].timestamp
            finish = self.chain[-1].timestamp

            average_time_consumed = round((finish - start) / (self.adjust_difficulty_blocks), 2) # 計算平均挖礦時間
            
            # 如果平均挖礦時間大於預設的區塊時間，降低難度
            if average_time_consumed > self.block_time:
                print(f"Average block time: {average_time_consumed}s. Lowering the difficulty")
                self.difficulty -= 1
            
            # 如果平均挖礦時間小於預設的區塊時間，提高難度
            else:
                print(f"Average block time: {average_time_consumed}s. Increasing the difficulty")
                self.difficulty += 1

    # Step5: Start the socket server and wait for other customers to connect.
    def start_socket_server(self):
        t = threading.Thread(target=self.wait_for_socket_connection)
        t.start()

    def wait_for_socket_connection(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: # 創建一個 socket 服務器
            s.bind((self.socket_host, self.socket_port)) # 綁定主機和端口
            s.listen() # 開始監聽連接

            while True:
                conn, address = s.accept() # 接受customer的連接
                
                # 創建一個線程來處理customer的消息
                client_handler = threading.Thread(
                    target=self.receive_socket_message,
                    args=(conn, address)
                )
                client_handler.start()

    # Step6: Receive and handle the message from customer.
    def receive_socket_message(self, connection, address):
        with connection:
            print(f'Connected by: {address}')
            while True:
                try:
                    message = connection.recv(1024) # 從連接中接收消息，最多1024字節
                    parsed_message = pickle.loads(message) # decode message -> dict{'request':..., 'date':...}
                    print("parsed_message: ", parsed_message) # 輸出解析後的消息
                except Exception:
                    pass
                if message:
                    if parsed_message["request"] == "get_balance":
                        print("Start to get the balance for client...")
                        address = parsed_message["address"] # 從解析後的消息中獲取customer的地址
                        balance = self.get_balance(address) # 獲取該地址的餘額

                        response = {
                            "reply": "get_balance", 
                            "data": balance
                        }
                        connection.send(pickle.dumps(response)) # 將回復消息序列化並發送給customer

                    elif parsed_message["request"] == "add_address":
                        print("Put client's address into database")

                        # 將現有地址列表中的地址提取出來，放入臨時列表中
                        temp_list=[]
                        for i in self.address_table:
                            temp_list.append(i[0])
                        self.address_table.append([parsed_message['data'], 0]) # 將customer發送的地址及其對應的餘額（這裡設為0）添加到地址表中
                        
                        response = {
                            "reply": "update_address",
                            "data":temp_list # return the current address table for customer.
                        }
                        connection.send(pickle.dumps(response))

                    elif parsed_message["request"] == "transaction":
                        print("Start to transaction for client...")
                        print("The message: ")
                        print(parsed_message)
                        address = parsed_message["address"] # 從解析後的消息中獲取地址
                        
                        # 調用初始化交易的函數，建立新的交易。
                        new_transaction = self.initialize_transaction(
                            parsed_message["address"], 
                            parsed_message["receiver"],
                            int(parsed_message["amount"]), 
                            int(parsed_message["fee"]),
                            parsed_message["comment"]
                        )

                        # 如果初始化交易函數返回的是字符串，表示交易被礦工拒絕，因為餘額不足。
                        if type(new_transaction) == type(""):
                            result_message = "The Transaction is rejected by miner, since y balance is not enough."
                        # 如果初始化交易成功，則將交易添加到交易池中。
                        else:
                            result_message = self.add_transaction(new_transaction, parsed_message["signature"])
                        
                        response = {
                            "reply": "transaction", 
                            "data": result_message
                        }
                        connection.send(pickle.dumps(response))

                    elif parsed_message["request"] == "apply":
                        # data : addr, amount
                        temp_list = parsed_message["data"] # 從解析後的消息中獲取地址和要增加的金額
                        for i in self.address_table:
                            # 如果地址表中存在客戶端的地址
                            if temp_list[0] == i[0]:
                                i[1] += temp_list[1]
                                break

                        # 打印更新後的地址表
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
                        connection.close() # 關閉與customer的連接
                        break
                    
                    else:
                        response = {
                            "message": "Unknown command."
                        }

    def get_balance(self, account):
        balance = 0
        # -----TODO-----
        ## Traverse the blockchain to get the balance
        for block in self.chain:
            # Check miner reward
            miner = False

            if block.miner == account:                # 如果區塊的礦工是指定的帳戶，則標記為礦工。
                miner = True
                balance += block.miner_rewards
            for transaction in block.transactions:
                if miner:                             # 如果是礦工，增加交易手續費到餘額。
                    balance += transaction.fee

                if transaction.sender == account:     # 如果交易的發送者是指定的帳戶，從餘額中減去轉出金額和交易手續費。
                    balance -= transaction.amounts
                    balance -= transaction.fee

                elif transaction.receiver == account: # 如果交易的接收者是指定的帳戶，增加接收金額到餘額。
                    balance += transaction.amounts

        # Add the test token that the customer had already applied
        for i in self.address_table:
            if account == i[0]:
                balance += i[1]
                break
    
        return balance

    def initialize_transaction(self, sender, receiver, amount, fee, message):
        # -----TODO-----
        # 使用get_balance確認交易起方有足夠的帳户餘完成這筆交易!
        ## 若帳戶餘額不足請回傳 "Balance not enough!"
        if self.get_balance(sender) < amount + fee:
            print("Balance not enough!")
            return False
        ## 若帳戶餘額足夠則回傳一個Transaction物件
        new_transaction = Transaction(sender, receiver, amount, fee, message)
        return new_transaction

    def add_transaction(self, transaction, signature):
        # 構建 RSA 公鑰
        public_key = '-----BEGIN RSA PUBLIC KEY-----\n'
        public_key += transaction.sender
        public_key += '\n-----END RSA PUBLIC KEY-----\n'
        public_key_pkcs = rsa.PublicKey.load_pkcs1(public_key.encode('utf-8'))

        transaction_str = self.transaction_to_string(transaction) # 將交易轉換為string形式
        
        # 檢查餘額是否足夠支付交易費用和金額
        if transaction.fee + transaction.amounts > self.get_balance(transaction.sender):
            print("Balance not enough!")
            return "Balance not enough!"
        try:
            # 驗證簽名是否有效
            rsa.verify(transaction_str.encode('utf-8'), signature, public_key_pkcs)
            print("Transaction was verified! It may be packed into block later.")

            self.pending_transactions.append(transaction) # 將交易添加到待處理交易列表中
            return True
        except Exception:
            print("RSA Verified wrong!")
            return "RSA Verified wrong!"

    # Step7: Generate the miner’s address and set the start function.
    def generate_address(self):
        public, private = rsa.newkeys(512) # 生成RSA公私鑰對，長度為512位。

        # 公私鑰轉成pkcs1格式
        public_key = public.save_pkcs1()
        private_key = private.save_pkcs1()

        return self.get_address_from_public(public_key), private_key

    def get_address_from_public(self, public):
        # 從公鑰中提取地址
        address = str(public).replace('\\n','')
        address = address.replace("b'-----BEGIN RSA PUBLIC KEY-----", '')
        address = address.replace("-----END RSA PUBLIC KEY-----'", '')
        address = address.replace(' ','')
        print('Address:', address)

        # 將miner的address存入table
        self.address_table.append([address, 0])
        return address

    def start(self):
        address, private = self.generate_address()
        self.create_genesis_block()

        if True:
            t = threading.Thread(target=self.interrup_control)
            t.start()

            # 循環挖礦，直到礦工停止挖礦。
            while(self.mining_flag == 1):
                # 在每一輪循環中，它挖掘一個新的區塊，然後調整挖礦難度。
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