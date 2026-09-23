import time
import pickle
import rsa
import socket
import threading
import os

# Step1: Define the class about the transaction
class Transaction:
    def __init__(self, sender, receiver, amounts, fee, message):
        self.sender = sender      # sender address
        self.receiver = receiver  # receiver address
        self.amounts = amounts    # cryptocurrency amount
        self.fee = fee            # transaction fee for the miner
        self.message = message
    
# Step2: Generate the address and Private Key
def generate_address():
    public, private = rsa.newkeys(512) # 生成RSA公私鑰對，長度為512位。

    # 將公私鑰轉換為PKCS1格式
    public_key = public.save_pkcs1()
    private_key = private.save_pkcs1()
    # The output is for learning the concept of the code, unnecessary while writting your HW.
    # print("\npublic_key = ",public_key, " , type is ", type(public_key),end="\n\n")
    # print("private_key = ", private_key, " , type is ", type(private_key),end="\n\n")
    return get_address_from_public(public_key), extract_from_private(private_key)

def get_address_from_public(public):
    address = str(public).replace('\\n','')
    address = address.replace("b'-----BEGIN RSA PUBLIC KEY-----", '')
    address = address.replace("-----END RSA PUBLIC KEY-----'", '')
    address = address.replace(' ','')
    return address

def extract_from_private(private):
    private_key = str(private).replace('\\n', '')
    private_key = private_key.replace("b'-----BEGIN RSA PRIVATE KEY-----", '')
    private_key = private_key.replace("-----END RSA PRIVATE KEY-----'", '')
    private_key = private_key.replace(' ', '')
    return private_key

# Step4: Write the handle_receive function.
def handle_receive():
    while True:
        try:
            response = client.recv(4096) # 接收來自customer的resopnse
            parsed_response = pickle.loads(response) # Unpickle response
            if response:
                # print(f"[*] Message from node: (parsed_response]")
                # 檢查response是否為空
                if parsed_response["reply"] == "update_address":
                    print("Successfully send the address to miner.")
                    print("Update local address table.")
                    temp_data = parsed_response["data"] # 取回更新的address data

                    for i in range(len(temp_data)):
                        address_table.append(temp_data[i]) # 更新用户列表
                        print(f"Node {i+1} : ", temp_data[i])

                elif parsed_response["reply"] == "get_balance":
                    print("Balance = ", parsed_response["data"])
                
                elif parsed_response["reply"] == "transaction":
                    print(parsed_response["data"])

        except Exception:
            print("ERROR !!!")
            print(f"[*] Message from miner: {parsed_response}")
            os._exit(0)

# Step5: Deploy and sign a transaction.
def initialize_transaction(sender, receiver, amount, fee, message):
    # -----ToDo-----
    ## Same to the work in miner.py , but no need to check balance !!!
    ## 請回傳一個Transaction物件
    new_transaction = Transaction(sender, receiver, amount, fee, message)
    return new_transaction

def transaction_to_string(transaction):
    transaction_dict = {
        'sender': str(transaction.sender),
        'receiver': str(transaction.receiver),
        'amounts': transaction.amounts,
        'fee': transaction.fee,
        'message': transaction.message
    }
    return str(transaction_dict)

def sign_transaction(transaction, private):
    # 建構private key string
    private_key = '-----BEGIN RSA PRIVATE KEY-----\n'
    private_key += private
    private_key += '\n-----END RSA PRIVATE KEY-----\n'
    private_key_pkcs = rsa.PrivateKey.load_pkcs1(private_key.encode('utf-8')) # Load private key
    transaction_str = transaction_to_string(transaction) 
    signature = rsa.sign(transaction_str.encode('utf-8'), private_key_pkcs, 'SHA-1') # 用private key做簽名
    return signature

# Step3: Write the main function.
# 執行此步驟時,可將上面程式碼註解
if __name__ == "__main__":
    # Test generate_address function
    # address, Private_key = generate_address()
    # print("address = ", address, " , type is ", type(address),end="\n\n") 
    # print("Private_key = ", Private_key, " , type is ", type(Private_key),end="\n\n")

    target_host = "127.0.0.1"
    target_port = 2024
    local_ip = "127.0.0.1"
    local_port = 2023
    address_table = []

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client.connect((target_host, target_port))
        print("Successfully connect to miner")
        receive_handler = threading.Thread(target=handle_receive, args=()) # 創建一個執行緒來處理receive
        receive_handler.start()

        command_dict = {
            "1": "generate_address",
            "2": "get_balance",
            "3": "transaction",
            "4": "close",
            "5": "apply"
        }

        while True:
            print("Command list:")
            print("1. generate_address")
            print("2. get_balance")
            print("3. transaction")
            print("4. close the application")
            print("5. apply some test tokens")
            command = input("Command: ")
            if str(command) not in command_dict.keys():
                print("Unknown command.")
                continue

            message = {"request":command_dict[str(command)]}

            if command_dict[str(command)] == "generate_address":
                address, private_key = generate_address()
                print(f"Address: {address}")
                print(f"Private key: {private_key}")
                message = {"request":"add_address", "data":address} # 建構要發送的消息，包含要添加的地址。
                client.send(pickle.dumps(message)) # pickle.dumps將message轉成binary然後由send發送

            elif command_dict[str(command)] == "get_balance":
                address = input("Address: ")
                message['address'] = address
                client.send(pickle.dumps(message))

            elif command_dict[str(command)] == "transaction":
                # 請求user輸入各種資訊
                message['address'] = input("Address: ")
                private_key = input("Private_key: ")
                message['receiver'] = input("Receiver: ")
                message['amount'] = input("Amount: ")
                message['fee'] = input("Fee: ")
                message['comment'] = input("Comment: ")

                # 初始化交易
                new_transaction = initialize_transaction(
                    message["address"],
                    message["receiver"],
                    int(message["amount"]),
                    int(message["fee"]),
                    message["comment"]
                )

                signature = sign_transaction(new_transaction, private_key) # 對交易進行簽名
                message["signature"] = signature # 添加簽名到message
                client.send(pickle.dumps(message))

            elif command_dict[str(command)] == "apply":
                addr = input("Address: ")
                amount = int(input("How many tokens you are going to apply : "))
                message['data'] = [addr, amount]
                client.send(pickle.dumps(message))

            elif command_dict[str(command)] == "close":
                client.send(pickle.dumps(message))
                # client.close()
                # print("!!!!")
                time.sleep(1)   # 等待子執行緒關閉
                os._exit(0)

            else:
                print("Unknown command.")
            time.sleep(1)

    except:
        print("The server has not yet been opened. Please try connecting again later")

a = generate_address()
print(a)