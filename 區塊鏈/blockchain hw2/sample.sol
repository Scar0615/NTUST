// SPDX-License-Identifier: MIT
pragma solidity 0.8.19;


// ---------------------------------------------------------------
// ERC20Interface 3-42
// ---------------------------------------------------------------
interface ERC20Interface {
    function totalSupply() external view returns (uint);
    function balanceOf(address tokenOwner) external view returns (uint balance);
    function allowance(address tokenOwner, address spender) external view returns (uint remaining);
    function transfer(address to, uint tokens) external returns (bool success);
    function approve(address spender, uint tokens) external returns (bool success);
    function transferFrom(address from, address to, uint tokens) external returns (bool success);
    event Transfer(address indexed from, address indexed to, uint tokens);
    event Approval(address indexed tokenOwner, address indexed spender, uint tokens);
}

// ---------------------------------------------------------------
// SafeMath library 3-44
// ---------------------------------------------------------------
contract SafeMath {
    function safeAdd (uint a, uint b) public pure returns (uint c) {
        c = a + b;
        require(c >= a);
    }

    function safeSub(uint a, uint b) public pure returns (uint c) {
        require(b <= a, "Invalid operation!");
        c = a - b;
    }

    function safeMul(uint a, uint b) public pure returns (uint c) {
        c = a * b;
        require(a==0 || c/a==b);
    }

    function safeDiv(uint a, uint b) public pure returns (uint c) {
        require(b > 0);
        c = a / b;
    }
}

// ---------------------------------------------------------------
// ERC20 Content 3-52~3-58
// ---------------------------------------------------------------
contract myToken is ERC20Interface, SafeMath{

    // 3 optional rules
    string public name;
    string public symbol;
    uint8 public decimals;
    uint256 public _totalSupply;

    mapping(address => uint256) balances;
    mapping(address => mapping(address => uint256)) allowed;

    constructor()
    {
        // optional rules
        name = "Culture Coin";  // Name of the token 
        symbol = "SepoliaETH";    // Symbol of the token
        decimals = 3;      // The minimum unit value of a token.

        _totalSupply = 500000000;
        balances[msg.sender] = _totalSupply;
        emit Transfer(address(0), msg.sender, _totalSupply);
        // faucet
        contractCreator = msg.sender;
    }

    // implement mandatory rules
    function totalSupply() public view override returns (uint)
    {
        return _totalSupply - balances[address(0)];
    }

    function balanceOf(address tokenOwner) public view override returns (uint balance)
    {
        return balances[tokenOwner];
    }

    function transfer(address to, uint tokens) public override returns (bool success)
    {
        balances[msg.sender] = safeSub(balances[msg.sender], tokens);
        balances[to] = safeAdd(balances[to], tokens);
        emit Transfer(msg.sender, to, tokens);
        return true;
    }

    function allowance(address tokenOwner, address spender) public view override returns (uint remaining)
    {
        return allowed[tokenOwner][spender];
    }

    function approve(address spender, uint tokens) public override returns (bool success)
    {
        allowed[msg.sender][spender] = tokens;
        emit Approval(msg.sender, spender, tokens);
        return true;
    }

    function transferFrom(address from, address to, uint tokens) public override returns (bool success)
    {
        balances[from] = safeSub(balances[from], tokens);
        allowed[from][msg.sender] = safeSub(allowed[from][msg.sender], tokens);
        balances[to] = safeAdd(balances[to], tokens);
        emit Transfer(from, to, tokens);
        return true;
    }

    // ---------------------------------------------------------------
    // faucet function
    // ---------------------------------------------------------------
    address public contractCreator;
    mapping(address => uint256) lastFaucet;

    function faucet(address from, address to) public returns (bool success) {
        require(from == contractCreator, "Only the contract creator can use the faucet.");
        require(block.timestamp > lastFaucet[to] + 30 seconds, "Please wait for 30 seconds before requesting again.");
        
        balances[from] = safeSub(balances[from], 500);
        balances[to] = safeAdd(balances[to], 500);
        emit Transfer(from, to, 500);
        
        lastFaucet[to] = block.timestamp;
        return true;
    }
}